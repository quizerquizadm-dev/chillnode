# syntax=docker/dockerfile:1
# RunPod serverless worker running UGC-VideoCaptioner-Abliterated
# (Qwen2.5-Omni-3B based, handles image + video + audio in one model).
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

# transformers==4.52.3 is the version the model card itself specifies —
# keep this pin exact. The rest are left loose since torch/runpod already
# ship in the base image or aren't version-sensitive for this use case;
# if the build fails on one of them, drop the pin for that package.
RUN pip install --no-cache-dir \
    runpod \
    transformers==4.52.3 \
    accelerate \
    soundfile \
    decord \
    qwen-omni-utils \
    requests

# Pre-download and cache the model weights into the image so a cold start
# only has to load them from local disk, not from the internet.
#
# This model repo is gated on Hugging Face — from_pretrained() needs an
# authenticated, access-approved token or it fails with exit code 1 before
# it ever downloads a byte. We mount the token as a BuildKit secret (not
# ENV/ARG) so it never gets baked into an image layer or build history.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python3 -c "\
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; \
Qwen2_5OmniForConditionalGeneration.from_pretrained('MahouOfficial/UGC-VideoCaptioner-Abliterated'); \
Qwen2_5OmniProcessor.from_pretrained('MahouOfficial/UGC-VideoCaptioner-Abliterated')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
