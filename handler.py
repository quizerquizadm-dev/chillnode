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
    "description": "...",
    "prompt":         "<optional admin instructions from Settings page>",
    "max_new_tokens": <optional int, 32-1024>
  }
}

Returns:
  {"caption": "...", "prompt_used": "..."}   on success
  {"error": "..."}     on failure (the dashboard treats this as a job
                        failure even though RunPod itself reports the
                        HTTP call as COMPLETED — see vision_worker.py's
                        handling of output.error)
"""

import os
import re
import subprocess
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


MAX_TOKENS_DEFAULT, MAX_TOKENS_MIN, MAX_TOKENS_MAX = 200, 32, 1024


def _render_prompt(template: str, media_type: str, title: str, description: str,
                   duration: str = "unknown") -> str:
    """Fill {media_type}/{title}/{description}/{duration} with plain string
    replacement (NOT str.format) so any other curly braces the admin types —
    JSON examples, etc. — can't crash the job with a KeyError."""
    out = template
    for key, value in (("{media_type}", media_type),
                       ("{title}", title),
                       ("{description}", description),
                       ("{duration}", duration)):
        out = out.replace(key, value)
    return out


def _video_duration_seconds(path: str):
    """Exact clip length via ffprobe (ships with the ffmpeg already installed
    in the Docker image). Returns float seconds, or None if it can't be read —
    a probe failure must never fail the caption job."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=20,
        )
        value = float(out.stdout.strip())
        return value if value > 0 else None
    except Exception:
        return None


def _format_duration(seconds: float) -> str:
    """12.4 -> '12 seconds', 1 -> '1 second', 65 -> '1 minute 5 seconds'."""
    total = int(round(seconds))
    if total < 1:
        return "less than 1 second"
    mins, secs = divmod(total, 60)
    parts = []
    if mins:
        parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
    if secs:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return " ".join(parts)


def _caption_mentions_duration(caption: str, seconds: float, duration_text: str) -> bool:
    """True if the model already worked the length into its description
    ('12 seconds', '12-second', '12s', or the total seconds of a longer clip)."""
    low = caption.lower()
    if duration_text.lower() in low:
        return True
    total = int(round(seconds))
    return bool(re.search(rf"\b{total}\s*-?\s*(s|sec|secs|second|seconds)\b", low))


def _clean_max_tokens(raw) -> int:
    try:
        return max(MAX_TOKENS_MIN, min(MAX_TOKENS_MAX, int(raw)))
    except (TypeError, ValueError):
        return MAX_TOKENS_DEFAULT


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

        # Admin-written instructions (Settings page -> bot_settings ->
        # vision_worker.py -> here). Blank/missing = built-in default.
        custom_prompt = (inp.get("prompt") or "").strip()
        template = custom_prompt or PROMPT_TEMPLATE
        # Videos: measure the real length so it can be put in the caption
        # (and used as {duration} inside the admin's prompt).
        duration_text = None
        duration_secs = None
        if media_type == "video":
            duration_secs = _video_duration_seconds(tmp_path)
            if duration_secs:
                duration_text = _format_duration(duration_secs)
        prompt_text = _render_prompt(template, media_type, title, description,
                                     duration_text or "unknown")
        # Works with the default AND any custom Settings prompt: tell the model
        # the real length and ask it to include it in the description itself.
        if duration_text:
            prompt_text += (
                f"\nThe video is exactly {duration_text} long. Mention this length "
                f"naturally as part of your description (for example: \"In this "
                f"{duration_text} clip, ...\")."
            )
        max_new_tokens = _clean_max_tokens(inp.get("max_new_tokens", MAX_TOKENS_DEFAULT))
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
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

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

                # Safety net: only if the model didn't mention the length itself.
        if duration_text and not _caption_mentions_duration(caption, duration_secs, duration_text):
            caption = f"This video is {duration_text} long. {caption}"

        # prompt_used is echoed back so you can verify in the RunPod console
        # (job -> Output) that your Settings-page instructions really arrived.
        return {
            "caption": caption,
            "prompt_used": prompt_text[:1500],
            "custom_prompt": bool(custom_prompt),
            "max_new_tokens": max_new_tokens,
            "video_length": duration_text,
        }

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


runpod.serverless.start({"handler": handler})
