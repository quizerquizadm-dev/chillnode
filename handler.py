"""
runpod_deploy/handler.py — RunPod serverless entrypoint.

Loaded ONCE per cold start (model stays warm across jobs while the
endpoint has an active worker; RunPod scales the worker back to zero
after the idle timeout you configure in the console — see the setup
instructions).

Expected input JSON (what the dashboard's vision_worker.py sends):
{
  "input": {
    "media_type": "image" | "video",
    "media_url":  "<https url the worker can download directly>",
    "title":       "...",
    "description": "..."
  }
}

Returns:
  {"caption": "..."}   on success
  {"error": "..."}     on failure (the dashboard treats this as a job
                        failure even though RunPod itself reports the
                        HTTP call as COMPLETED — see vision_worker.py's
                        handling of output.error)
"""

import os
import tempfile
import traceback

import requests
import runpod
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it"

print("[handler] Loading model — this only happens once per cold start...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("[handler] Model loaded and ready.")

# NOTE: Qwen3-VL is image+video only — no audio understanding. If a video
# has meaningful dialogue/sound your captions need to reflect, that has to
# come from a separate speech-to-text pass; this model will silently just
# describe the visuals.

PROMPT_TEMPLATE = (
    "Describe this {media_type} in 1-2 sentences covering the subject, setting, mood, "
    "and any key visual details. Be specific and factual. This description will be used "
    "by a chatbot to decide when to naturally share this content in conversation.\n"
    "Title: \"{title}\"\n"
    "Admin-provided description: \"{description}\""
)


def _download_to_tempfile(url: str, suffix: str) -> str:
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
        return f.name


def handler(event):
    inp = event.get("input") or {}
    media_type = inp.get("media_type")
    media_url = inp.get("media_url")
    title = inp.get("title") or ""
    description = inp.get("description") or "none provided"

    if media_type not in ("image", "video"):
        return {"error": f"media_type must be 'image' or 'video', got: {media_type!r}"}
    if not media_url:
        return {"error": "media_url is required"}

    tmp_path = None
    try:
        suffix = ".mp4" if media_type == "video" else ".jpg"
        tmp_path = _download_to_tempfile(media_url, suffix)

        prompt_text = PROMPT_TEMPLATE.format(
            media_type=media_type, title=title, description=description
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": media_type, media_type: tmp_path},
                {"type": "text", "text": prompt_text},
            ],
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200)

        # Trim off the echoed input tokens so we're left with just the
        # newly generated caption (the standard Qwen-VL decode pattern).
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        if not caption:
            return {"error": "Model returned an empty caption"}

        return {"caption": caption}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


runpod.serverless.start({"handler": handler})
