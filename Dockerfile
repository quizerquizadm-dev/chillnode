# syntax=docker/dockerfile:1
# RunPod serverless worker running Qwen3-VL-8B-Abliterated-Caption-it
# (Qwen3-VL-8B based, handles image + video captioning, uncensored).
# No audio understanding — this is a VL model, not an Omni model.
#
# Build & push:
#   docker build -t <your-dockerhub-username>/vision-worker:latest .
#   docker push <your-dockerhub-username>/vision-worker:latest
#
# The model weights are baked into the image at build time on purpose —
# this avoids a multi-GB download on every cold start, which matters a
# lot when the endpoint scales to zero between uploads (see the main
# setup instructions for why that's the whole point of this design).

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# transformers==4.57.6 is required — Qwen3VLForConditionalGeneration does
# not exist yet in the 4.52.x line this project started on. qwen-vl-utils
# (not qwen-omni-utils) is the correct helper package for VL-family models;
# it's what handles frame sampling for video input.
RUN pip install --no-cache-dir \
    runpod \
    transformers==4.57.6 \
    accelerate \
    decord \
    qwen-vl-utils \
    requests

# Pre-download and cache the model weights into the image so a cold start
# only has to load them from local disk, not from the internet.
#
# This model repo is gated on Hugging Face (instant-accept: log in, click
# Agree — not a manual-review queue) — from_pretrained() needs an
# authenticated, access-approved token or it fails before downloading a
# byte. We mount the token as a BuildKit secret (not ENV/ARG) so it never
# gets baked into an image layer or build history.
RUN --mount=type=secret,id=hf_token \
    sh -c 'test -s /run/secrets/hf_token && echo "[hf_token] secret received, length=$(wc -c < /run/secrets/hf_token)" || echo "[hf_token] secret MISSING or empty — check the HF_TOKEN repo secret and the secrets: block in build.yml"' && \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it', \
    token='$HF_TOKEN', \
    ignore_patterns=['*.bin', '*.pt', '*.onnx', '*.h5'], \
)"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
