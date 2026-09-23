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
import subprocess
import tempfile
import time
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


def _normalize_video(input_path: str) -> str:
    """Re-mux the downloaded video with explicit, unambiguous color-space
    metadata before it ever reaches PyAV/torchvision's frame decoder.

    Videos that come from the admin dashboard have been through Telegram's
    own transcoder first (upload -> Telegram -> getFile -> this handler).
    Telegram's re-encode commonly tags the file as `csp:bt709` but leaves
    color primaries/transfer characteristics UNSET ("prim:unknown
    trc:unknown"). When ffmpeg's swscale later has to build a conversion
    graph from that ambiguous format to rgb24 for the model, it can
    intermittently fail with:

        av.error.BlockingIOError: [Errno 11] Resource temporarily
        unavailable; [swscaler] Failed initializing scaling graph

    Videos fetched from a URL that was never touched by Telegram (e.g. a
    plain hosted .mp4 used for manual testing) typically already have full
    color metadata, which is why manual tests via the RunPod "Requests"
    tab don't reproduce this — the failure is specific to Telegram-relayed
    uploads, not the handler logic itself.

    Explicitly stamping bt709 on all three fields removes the ambiguity
    up front, so this failure mode can't occur regardless of what the
    source encoder did or didn't set. Audio is dropped since this is a
    VL (not Omni) model and doesn't use it, which also speeds up the pass.
    """
    normalized_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-movflags", "+faststart",
        "-threads", "1",
        normalized_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180
        )
        if result.returncode != 0 or not os.path.exists(normalized_path) \
                or os.path.getsize(normalized_path) == 0:
            raise RuntimeError(f"ffmpeg normalization failed: {result.stderr[-1000:]}")
        return normalized_path
    except Exception:
        # Best-effort cleanup of a partial/failed output before re-raising
        # so we don't leak a broken temp file if something downstream
        # decides to retry with the original instead.
        if os.path.exists(normalized_path):
            os.remove(normalized_path)
        raise


def _run_vision_pipeline(messages, max_new_tokens: int, attempts: int = 3):
    """Runs the decode -> generate -> decode pipeline with a short retry
    loop. This is a backstop for any transient decode error that survives
    _normalize_video (e.g. a genuinely flaky swscale EAGAIN unrelated to
    color metadata, or an image-path issue) — it is not the primary fix,
    _normalize_video is, but it means a single hiccup never fails the job.
    """
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device).to(model.dtype)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            return caption
        except Exception as e:
            last_err = e
            transient = (
                "Resource temporarily unavailable" in str(e)
                or "BlockingIOError" in type(e).__name__
                or "scaling graph" in str(e)
            )
            if attempt < attempts and transient:
                time.sleep(1.5 * attempt)  # 1.5s, 3s backoff
                continue
            raise
    raise last_err


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

    # Admin dashboard may override the prompt/token-budget per media type
    # (see vision_worker.py's _build_runpod_input) — fall back to our own
    # defaults when it doesn't.
    prompt = inp.get("prompt") or PROMPT_TEMPLATE
    max_new_tokens = inp.get("max_new_tokens") or 200

    tmp_path = None
    normalized_path = None
    try:
        suffix = ".mp4" if media_type == "video" else ".jpg"
        tmp_path = _download_to_tempfile(media_url, suffix)

        media_path = tmp_path
        if media_type == "video":
            # Fixes the intermittent swscale EAGAIN failure on
            # Telegram-relayed uploads — see _normalize_video docstring.
            normalized_path = _normalize_video(tmp_path)
            media_path = normalized_path

        prompt_text = prompt.format(
            media_type=media_type, title=title, description=description
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": media_type, media_type: media_path},
                {"type": "text", "text": prompt_text},
            ],
        }]

        caption = _run_vision_pipeline(messages, max_new_tokens=max_new_tokens)

        if not caption:
            return {"error": "Model returned an empty caption"}

        return {"caption": caption}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if normalized_path and os.path.exists(normalized_path):
            os.remove(normalized_path)


runpod.serverless.start({"handler": handler})"""
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
