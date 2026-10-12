"""Resume OCR demo server: render, OCR and extract a resume, streaming each stage to the page.

POST /extract (multipart, either `sample=<id>` or `file=<pdf|png|jpg>`) answers with a
server-sent-event stream of JSON messages:
  {"type": "pages", "pages": [{"image": <data URL>, "width", "height"}], "total_pages", "render_ms"}
  {"type": "ocr", "page": i, "lines": [{"text", "box", "words": [{"text", "box", "confidence"}]}], "ms"}
  {"type": "delta", "text": <next piece of the JSON document as the model writes it>}
  {"type": "result", "data": <resume JSON>, "ms", "completion_tokens"}
  {"type": "error", "message"}
Boxes are [x0, y0, x1, y1] relative to the page. Uploads are processed in memory and never stored.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image

from extract import Delta, Done, Extractor
from ocr import Line, Ocr, UnsupportedFile, page_text, render_pages
from samples import Sample, build_samples

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8080"))
LLM_MODEL = os.environ.get("RESUME_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8")
# Set to an OpenAI-compatible base URL to use an existing model server instead of starting vLLM.
LLM_URL = os.environ.get("RESUME_LLM_URL", "")
VLLM_PORT = 8000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PAGES = 2
# Jobs running at once. vLLM batches concurrent requests well; this bounds memory and keeps
# one visitor's queue wait short when several people press the button together.
MAX_CONCURRENT_JOBS = 4
RATE_LIMIT_JOBS = 15
RATE_LIMIT_WINDOW_S = 600

log = logging.getLogger("resume-ocr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
state: dict[str, Any] = {}
recent_jobs: dict[str, deque[float]] = defaultdict(deque)


def start_vllm() -> subprocess.Popen[bytes]:
    cmd = [
        "vllm", "serve", LLM_MODEL,
        "--host", "127.0.0.1",
        "--port", str(VLLM_PORT),
        "--max-model-len", "16384",
        # Leaves room on a 24 GB GPU for the OCR models, which load after vLLM sizes its cache.
        "--gpu-memory-utilization", "0.72",
        # xgrammar only honors this server-wide. Without it, the JSON grammar allows any whitespace
        # between tokens and a small model can spend its whole token budget on newlines.
        "--structured-outputs-config", '{"backend": "xgrammar", "disable_any_whitespace": true}',
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


async def exit_if_dead(proc: subprocess.Popen[bytes]) -> None:
    # Without its model server the replica can only fail requests; exiting lets it restart.
    while proc.poll() is None:
        await asyncio.sleep(2)
    log.error("vLLM exited with %s", proc.returncode)
    os._exit(1)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    proc = None
    base_url = LLM_URL
    if not base_url:
        proc = start_vllm()
        base_url = f"http://127.0.0.1:{VLLM_PORT}/v1"
        state["watchdog"] = asyncio.create_task(exit_if_dead(proc))
    extractor = Extractor(base_url, LLM_MODEL)
    t0 = time.monotonic()
    while not await extractor.ready():
        await asyncio.sleep(2)
    log.info("model server ready after %.0fs", time.monotonic() - t0)
    state["extractor"] = extractor
    state["ocr"] = await asyncio.to_thread(Ocr)
    log.info("OCR ready on %s", state["ocr"].device)
    samples = await asyncio.to_thread(build_samples)
    state["samples"] = {s.id: s for s in samples}
    state["thumbs"] = {s.id: await asyncio.to_thread(thumbnail, s) for s in samples}
    state["jobs"] = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    # uvicorn accepts no connections until this yields, so the replica is not ready before
    # both models are loaded and warm.
    yield
    if proc is not None:
        proc.terminate()


app = FastAPI(lifespan=lifespan)


def thumbnail(sample: Sample) -> bytes:
    pages, _ = render_pages(sample.pdf, 1)
    img = pages[0]
    img.thumbnail((240, 320))
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def data_url(img: Image.Image) -> str:
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def allow(ip: str) -> bool:
    now = time.monotonic()
    times = recent_jobs[ip]
    while times and now - times[0] > RATE_LIMIT_WINDOW_S:
        times.popleft()
    if len(times) >= RATE_LIMIT_JOBS:
        return False
    times.append(now)
    return True


def sse(msg: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(msg)}\n\n".encode()


def line_json(line: Line) -> dict[str, Any]:
    return {
        "text": line.text,
        "box": [round(v, 4) for v in line.box],
        "words": [
            {"text": w.text, "box": [round(v, 4) for v in w.box], "confidence": round(w.confidence, 3)}
            for w in line.words
        ],
    }


async def run_job(data: bytes) -> AsyncIterator[bytes]:
    ocr: Ocr = state["ocr"]
    extractor: Extractor = state["extractor"]
    try:
        async with asyncio.timeout(30):
            await state["jobs"].acquire()
    except TimeoutError:
        yield sse({"type": "error", "message": "The demo is busy right now. Try again in a moment."})
        return
    try:
        t0 = time.monotonic()
        try:
            pages, total = await asyncio.to_thread(render_pages, data, MAX_PAGES)
        except UnsupportedFile as e:
            yield sse({"type": "error", "message": f"Unsupported file: {e}."})
            return
        yield sse(
            {
                "type": "pages",
                "pages": [{"image": data_url(p), "width": p.width, "height": p.height} for p in pages],
                "total_pages": total,
                "render_ms": round((time.monotonic() - t0) * 1000),
            }
        )
        all_lines = []
        for i, page in enumerate(pages):
            t1 = time.monotonic()
            lines = await asyncio.to_thread(ocr.read, page)
            all_lines.append(lines)
            yield sse(
                {
                    "type": "ocr",
                    "page": i,
                    "lines": [line_json(ln) for ln in lines],
                    "ms": round((time.monotonic() - t1) * 1000),
                }
            )
        text = page_text(all_lines)
        if not text.strip():
            yield sse({"type": "error", "message": "No text found on the page."})
            return
        t2 = time.monotonic()
        async for event in extractor.stream(text):
            if isinstance(event, Delta):
                yield sse({"type": "delta", "text": event.text})
            elif isinstance(event, Done):
                yield sse(
                    {
                        "type": "result",
                        "data": event.data,
                        "ms": round((time.monotonic() - t2) * 1000),
                        "completion_tokens": event.completion_tokens,
                    }
                )
    except Exception:
        log.exception("job failed")
        yield sse({"type": "error", "message": "Extraction failed. Try another file."})
    finally:
        state["jobs"].release()


@app.post("/extract")
async def extract(request: Request, sample: str = Form(""), file: UploadFile | None = None) -> Response:
    if not allow(client_ip(request)):
        return JSONResponse({"error": "Too many requests. Try again in a few minutes."}, status_code=429)
    if sample:
        found = state["samples"].get(sample)
        if found is None:
            return JSONResponse({"error": "Unknown sample."}, status_code=404)
        data = found.pdf
    elif file is not None:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "Files up to 5 MB, please."}, status_code=413)
    else:
        return JSONResponse({"error": "Send a sample id or a file."}, status_code=400)
    return StreamingResponse(
        run_job(data),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/samples")
async def list_samples() -> JSONResponse:
    return JSONResponse([{"id": s.id, "title": s.title, "blurb": s.blurb} for s in state["samples"].values()])


@app.get("/samples/{sample_id}.png")
async def sample_thumb(sample_id: str) -> Response:
    thumb = state["thumbs"].get(sample_id)
    if thumb is None:
        return Response(status_code=404)
    return Response(thumb, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/samples/{sample_id}.pdf")
async def sample_pdf(sample_id: str) -> Response:
    found = state["samples"].get(sample_id)
    if found is None:
        return Response(status_code=404)
    return Response(found.pdf, media_type="application/pdf")


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({"ok": True, "model": LLM_MODEL, "ocr_device": state["ocr"].device})


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.get("/app.js")
async def app_js() -> FileResponse:
    return FileResponse(HERE / "app.js", media_type="text/javascript")


@app.get("/app.css")
async def app_css() -> FileResponse:
    return FileResponse(HERE / "app.css", media_type="text/css")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
