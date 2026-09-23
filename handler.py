"""
runpod_deploy/handler.py — RunPod serverless entrypoint.

Loaded ONCE per cold start (model stays warm across jobs while the
endpoint has an active worker; RunPod scales the worker back to zero
after the idle timeout you configure in the console — see the setup
instructions).

Model: prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it — a Qwen3-VL-8B
fine-tune for uncensored image/video captioning. This is a VL model,
not an Omni model: it does not understand audio.

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

PROMPT_TEMPLATE = (
    "Describe this {media_type} in 1-2 sentences covering the subject, setting, mood, "
    "and any key visual details. Be specific and factual. This description will be used "
    "by a chatbot to decide when to naturally share this content in conversation.\n"
    "Title: \"{title}\"\n"
    "Admin-provided description: \"{description}\""
)


from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Some hosts (e.g. Wikimedia) reject requests with the default
# python-requests User-Agent and return 403 Forbidden. A normal
# browser-like UA is accepted almost everywhere, including
# Telegram's api.telegram.org/file/... URLs.
_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

_session = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=1.5,          # 0s, 1.5s, 3s between retries
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
_session.mount("https://", HTTPAdapter(max_retries=_retry))
_session.mount("http://", HTTPAdapter(max_retries=_retry))


def _download_to_tempfile(url: str, suffix: str) -> str:
    resp = _session.get(
        url,
        timeout=(10, 60),   # (connect timeout, read timeout)
        stream=True,
        headers=_DOWNLOAD_HEADERS,
        allow_redirects=True,
    )
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
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

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # BatchFeature.to() only casts floating-point tensors (pixel_values,
        # etc.) to the given dtype and leaves integer tensors (input_ids,
        # attention_mask) untouched, so this is safe to chain.
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200)

        # Slice off the input prompt tokens so we only decode the newly
        # generated caption, not the echoed-back prompt.
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
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
