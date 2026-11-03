"""Step 1 — Load original scanned / unhighlighted invoice PDF as page images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf


def load_pdf_pages(pdf_path: str | Path, dpi: int = 200) -> list[np.ndarray]:
    """Render each PDF page to a BGR numpy image (OpenCV-compatible)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(pdf_path)
    pages: list[np.ndarray] = []
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            pages.append(img[:, :, ::-1].copy())  # RGB → BGR
    finally:
        doc.close()

    if not pages:
        raise ValueError(f"No pages found in PDF: {pdf_path}")

    return pages


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Native PDF text layer (empty string if scan-only / no text)."""
    pdf_path = Path(pdf_path)
    doc = pymupdf.open(pdf_path)
    try:
        parts = [(page.get_text("text") or "").strip() for page in doc]
    finally:
        doc.close()
    return "\n\n".join(p for p in parts if p)
