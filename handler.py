"""
handler.py — RunPod serverless entrypoint.

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
    "prompt":      "<optional> admin-written instructions; may contain the
                    placeholders {media_type}, {title}, {description}.
                    Empty/missing -> the built-in default prompt below.",
    "max_new_tokens": <optional int, clamped to 32-1024, default 200>
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

# Printed FIRST, before any heavy import, so the RunPod logs always show
# which build is running (GIT_SHA is baked in by the Dockerfile/build.yml).
print(f"[handler] build={os.environ.get('GIT_SHA', 'unknown')[:7]} starting", flush=True)

# The model weights are baked into the image, so never call out to the
# Hugging Face Hub at runtime (the repo is gated; an online check without a
# token can fail even though the files are already on disk). Set
# HF_HUB_OFFLINE=0 on the endpoint if you ever want to override this.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import tempfile
import traceback

import requests
import runpod
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it"

print("[handler] Loading model — this only happens once per cold start...", flush=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("[handler] Model loaded and ready.", flush=True)

# Used ONLY when the admin has not written their own instructions in the
# dashboard (Settings -> Vision description prompts). Placeholders are
# replaced literally, so any other braces in an admin's text are safe.
DEFAULT_PROMPT_TEMPLATE = (
    "Describe this {media_type} in 1-2 sentences covering the subject, setting, mood, "
    "and any key visual details. Be specific and factual. This description will be used "
    "by a chatbot to decide when to naturally share this content in conversation.\n"
    "Title: \"{title}\"\n"
    "Admin-provided description: \"{description}\""
)

MAX_PROMPT_CHARS = 4000
DEFAULT_MAX_NEW_TOKENS = 200


def _render_prompt(template: str, media_type: str, title: str, description: str) -> str:
    return (
        template
        .replace("{media_type}", media_type)
        .replace("{title}", title)
        .replace("{description}", description)
    )


def _clamp_tokens(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_MAX_NEW_TOKENS
    return max(32, min(1024, n))


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

        custom = inp.get("prompt")
        use_custom = isinstance(custom, str) and custom.strip() != ""
        template = custom.strip()[:MAX_PROMPT_CHARS] if use_custom else DEFAULT_PROMPT_TEMPLATE
        prompt_text = _render_prompt(template, media_type, title, description)
        max_new_tokens = _clamp_tokens(inp.get("max_new_tokens"))
        print(f"[handler] job {media_type}: prompt={'admin' if use_custom else 'default'}, "
              f"max_new_tokens={max_new_tokens}", flush=True)

        messages = [{
            "role": "user",
            "content": [
                {"type": media_type, media_type: tmp_path},
                {"type": "text", "text": prompt_text},
            ],
        }]

        # transformers (>=4.57) loads/samples the image or video itself, so
        # qwen-vl-utils is no longer needed on this path.
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        # .to(device).to(dtype) only casts floating-point tensors
        # (pixel_values, ...) and leaves integer ones (input_ids,
        # attention_mask) untouched.
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

        return {"caption": caption}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


runpod.serverless.start({"handler": handler})
