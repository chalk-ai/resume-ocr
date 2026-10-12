#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the resume OCR demo as a GPU ScalingGroup.

One replica runs both models on one GPU: docTR for OCR, and Qwen3-4B-Instruct behind vLLM for
structured extraction. The replica also serves the demo page. The image builds on the official
vLLM image; the LLM's weights download when a replica starts.

Credentials come from `chalk config` in the current directory (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>] [--gpu nvidia-l4]
Then open the printed URL and pick a resume.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# FP8 weights halve the bytes read per token, and decoding on an L4 is bound by memory bandwidth,
# so this streams roughly twice as fast as the bf16 checkpoint.
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FP8"
VLLM_VERSION = "v0.31.0"
PIP_PACKAGES = ["python-doctr==1.1.0", "pypdfium2==5.14.0", "reportlab==5.0.1", "python-multipart==0.0.32"]
APP_FILES = ["server.py", "ocr.py", "extract.py", "samples.py", "index.html", "app.js", "app.css"]


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="resume-ocr")
    # Defaults to the environment in `chalk config` (or CHALK_ENVIRONMENT_ID).
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--gpu", default="nvidia-l4")
    parser.add_argument("--ready-timeout", type=float, default=2400.0)
    args = parser.parse_args()

    load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup, StartupProbe

    image = (
        # vLLM, torch and CUDA come prebuilt; the build only adds small layers on top.
        Image.base(f"vllm/vllm-openai:{VLLM_VERSION}")
        # OpenCV, used by docTR, needs these.
        .apt_install(["libgl1", "libglib2.0-0"])
        # Pin torch and torchvision to the base image's own builds so pip cannot replace them.
        .run_commands(
            'python3 -c "import torch, torchvision; '
            + "print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')\""
            + " > /tmp/torch-constraints.txt",
            "python3 -m pip install --no-cache-dir -c /tmp/torch-constraints.txt " + " ".join(PIP_PACKAGES),
        )
        .env({"DOCTR_CACHE_DIR": "/models/doctr"})
        # The OCR weights are small enough to bake in. The LLM's 8 GB are fetched by vLLM at boot.
        .run_commands(
            'python3 -c "from doctr.models import ocr_predictor; '
            + "ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)\""
        )
    )
    # The app files total more than the 32 KiB that strategy="copy" inlines, so they ship on a
    # volume; every file under /app must use the same strategy or the mount hides the others.
    for name in APP_FILES:
        image = image.add_local_file(str(HERE / name), f"/app/{name}", strategy="volume")
    image = image.workdir("/app")

    sg = ScalingGroup(
        image=image,
        name=args.name,
        gpu=args.gpu,
        cpu="4",
        memory="24Gi",
        env={
            "RESUME_LLM_MODEL": LLM_MODEL,
            "PORT": "8080",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_NO_USAGE_STATS": "1",
        },
        entrypoint=["python3", "/app/server.py"],
        port=8080,
        # /status answers only once both models are loaded. Without these probes a rollout
        # retires the old replica while the new one is still loading vLLM, and deploy() returns
        # before the URL serves anything.
        readiness_probe=ReadinessProbe.http("/status", period_seconds=10, failure_threshold=3),
        startup_probe=StartupProbe.http("/status"),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=3000.0, ready_timeout=args.ready_timeout)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
