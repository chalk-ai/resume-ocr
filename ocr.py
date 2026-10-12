"""Page rendering and OCR: PDFs and images in, words with page-relative boxes out."""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

# 150 DPI keeps small resume type legible to the recognizer without oversized page images.
RENDER_SCALE = 150 / 72
MAX_IMAGE_SIDE = 2200
# A gutter must have at least this many lines on each side to count as a column split.
MIN_LINES_PER_COLUMN = 5

Image.MAX_IMAGE_PIXELS = 40_000_000
# PDFium is not thread-safe: two documents rendered from parallel threads can deadlock the process.
# Rendering takes milliseconds, so serializing it costs concurrent jobs nothing noticeable.
_PDFIUM_LOCK = threading.Lock()


class UnsupportedFile(ValueError):
    pass


@dataclass(frozen=True)
class Word:
    text: str
    # (x0, y0, x1, y1), each in [0, 1] relative to the page.
    box: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class Line:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (
            min(w.box[0] for w in self.words),
            min(w.box[1] for w in self.words),
            max(w.box[2] for w in self.words),
            max(w.box[3] for w in self.words),
        )


def render_pages(data: bytes, max_pages: int) -> tuple[list[Image.Image], int]:
    """Returns up to `max_pages` RGB page images and the document's total page count."""
    if data[:5] == b"%PDF-":
        with _PDFIUM_LOCK:
            try:
                pdf = pdfium.PdfDocument(data)
            except pdfium.PdfiumError as e:
                raise UnsupportedFile(f"could not open the PDF ({e})") from e
            total = len(pdf)
            pages = [pdf[i].render(scale=RENDER_SCALE).to_pil().convert("RGB") for i in range(min(total, max_pages))]
            pdf.close()
        return pages, total
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (OSError, Image.DecompressionBombError) as e:
        raise UnsupportedFile("expected a PDF, PNG or JPEG") from e
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return [img], 1


class Ocr:
    """docTR text detection + recognition. One page batch at a time per process."""

    def __init__(self) -> None:
        super().__init__()
        import torch
        from doctr.models import ocr_predictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
            assume_straight_pages=True,
        ).to(self.device)
        self.lock = threading.Lock()
        self.read(Image.new("RGB", (850, 1100), "white"))

    def read(self, page: Image.Image) -> list[Line]:
        with self.lock:
            doc = self.model([np.asarray(page)])
        lines: list[Line] = []
        for block in doc.pages[0].blocks:
            for line in block.lines:
                words = [
                    Word(
                        text=w.value,
                        box=(w.geometry[0][0], w.geometry[0][1], w.geometry[1][0], w.geometry[1][1]),
                        confidence=float(w.confidence),
                    )
                    for w in line.words
                    if w.value.strip()
                ]
                if words:
                    lines.append(Line(words))
        return reading_order(lines)


def _column_split(lines: list[Line]) -> float | None:
    """An x position no line crosses, with enough lines on both sides to be two real columns."""
    for step in range(20, 81):
        x = step / 100
        left = sum(1 for ln in lines if ln.box[2] <= x)
        right = sum(1 for ln in lines if ln.box[0] >= x)
        if left + right == len(lines) and min(left, right) >= MIN_LINES_PER_COLUMN:
            return x
    return None


def reading_order(lines: list[Line]) -> list[Line]:
    """Top-to-bottom, and for a two-column page, the whole left column before the right one.

    The detector returns lines ordered by height on the page, which interleaves a sidebar with
    the main column; a language model reading that text sees sentences spliced together.
    """
    by_row = sorted(lines, key=lambda ln: (round(ln.box[1], 3), ln.box[0]))
    split = _column_split(lines)
    if split is None:
        return by_row
    return [ln for ln in by_row if ln.box[2] <= split] + [ln for ln in by_row if ln.box[0] >= split]


def page_text(pages: list[list[Line]]) -> str:
    out = []
    for i, lines in enumerate(pages):
        if len(pages) > 1:
            out.append(f"--- page {i + 1} ---")
        out.extend(ln.text for ln in lines)
    return "\n".join(out)
