"""Orchestrates upload steps 1–3 and writes artifacts under uploads/."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import cv2

from services.ocr import run_ocr
from services.pdf_loader import load_pdf_pages
from services.preprocess import preprocess_image

ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = ROOT / "uploads"

UPLOADS_DIR.mkdir(exist_ok=True)


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return cleaned or "invoice"


def process_pdf_bytes(data: bytes, stem: str = "invoice") -> dict:
    """PDF → preprocess → OCR; persist under uploads/<job_id>/."""
    job_id = f"{stem}_{uuid.uuid4().hex[:8]}"
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / "input.pdf"
    pdf_path.write_bytes(data)

    pages = load_pdf_pages(pdf_path)
    for i, page in enumerate(pages):
        cv2.imwrite(str(job_dir / f"page{i}_original.png"), page)

    page_results = []
    for i, page in enumerate(pages):
        processed = preprocess_image(page)
        cv2.imwrite(str(job_dir / f"page{i}_preprocessed.png"), processed)

        ocr_out = run_ocr(processed)
        page_payload = {"page_index": i, **ocr_out}
        page_results.append(page_payload)
        (job_dir / f"page{i}_ocr.json").write_text(
            json.dumps(page_payload, indent=2),
            encoding="utf-8",
        )

    handoff = {
        "job_id": job_id,
        "source_stem": stem,
        "page_count": len(pages),
        "upload_dir": str(job_dir),
        "pages": page_results,
    }
    (job_dir / "ocr_handoff.json").write_text(
        json.dumps(handoff, indent=2),
        encoding="utf-8",
    )
    return handoff
