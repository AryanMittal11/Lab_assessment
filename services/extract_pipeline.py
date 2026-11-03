"""Extract process: load OCR handoff → per-page extract → persist."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from services.extract import extract_from_ocr as extract_rules
from services.llm_groq import extract_from_ocr as extract_groq
from services.pdf_loader import extract_pdf_text

ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = ROOT / "uploads"

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")
_VALID_MODES = frozenset({"rules", "groq"})


def extract_mode() -> str:
    """EXTRACT_MODE=rules|groq (default rules)."""
    mode = (os.getenv("EXTRACT_MODE") or "rules").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"EXTRACT_MODE must be one of {sorted(_VALID_MODES)}, got {mode!r}")
    return mode


def _extractor() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return extract_groq if extract_mode() == "groq" else extract_rules


def _job_dir(job_id: str) -> Path:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError("Invalid job_id")
    path = UPLOADS_DIR / job_id
    if not path.is_dir():
        raise FileNotFoundError(f"Unknown job_id: {job_id}")
    return path


def load_ocr_handoff(job_id: str) -> dict[str, Any]:
    handoff_path = _job_dir(job_id) / "ocr_handoff.json"
    if not handoff_path.is_file():
        raise FileNotFoundError(f"No ocr_handoff.json for job_id: {job_id}")
    return json.loads(handoff_path.read_text(encoding="utf-8"))


def _resolve_pdf_text(page: dict[str, Any], handoff: dict[str, Any] | None, job_dir: Path | None) -> str:
    for candidate in (
        page.get("pdf_text"),
        (handoff or {}).get("pdf_text"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    if job_dir is not None:
        pdf_path = job_dir / "input.pdf"
        if pdf_path.is_file():
            try:
                return extract_pdf_text(pdf_path)
            except Exception:
                return ""
    return ""


def extract_pages(
    pages: list[dict[str, Any]],
    job_dir: Path | None = None,
    handoff: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run configured extractor on each page OCR dict; optionally write page{{i}}_extract.json."""
    extract_fn = _extractor()
    results: list[dict[str, Any]] = []
    for page in pages:
        page_index = int(page.get("page_index", len(results)))
        ocr_out = {
            "text_blocks": page.get("text_blocks") or [],
            "full_text": page.get("full_text") or "",
            "layout_rows": page.get("layout_rows") or [],
            "tables": page.get("tables") or [],
            "pdf_text": _resolve_pdf_text(page, handoff, job_dir),
        }
        extracted = extract_fn(ocr_out)
        payload = {"page_index": page_index, **extracted}
        results.append(payload)
        if job_dir is not None:
            (job_dir / f"page{page_index}_extract.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
    return results


def process_job(job_id: str) -> dict[str, Any]:
    """Talk to upload via shared job folder: read ocr_handoff.json, write extract artifacts."""
    job_dir = _job_dir(job_id)
    handoff = load_ocr_handoff(job_id)
    pages = handoff.get("pages") or []
    mode = extract_mode()
    extractions = extract_pages(pages, job_dir=job_dir, handoff=handoff)

    result = {
        "job_id": job_id,
        "extract_mode": mode,
        "page_count": len(extractions),
        "upload_dir": str(job_dir),
        "pages": extractions,
    }
    (job_dir / "extract_handoff.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def process_ocr_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Direct OCR→extract without job folder.
    Accepts either a full handoff {pages:[...]} or a single page OCR dict.
    """
    mode = extract_mode()
    if "pages" in payload and isinstance(payload["pages"], list):
        pages = payload["pages"]
        job_id = payload.get("job_id")
        return {
            "job_id": job_id,
            "extract_mode": mode,
            "page_count": len(pages),
            "pages": extract_pages(pages, job_dir=None, handoff=payload),
        }

    # single page OCR dict
    extracted = _extractor()(
        {
            "text_blocks": payload.get("text_blocks") or [],
            "full_text": payload.get("full_text") or "",
            "layout_rows": payload.get("layout_rows") or [],
            "tables": payload.get("tables") or [],
            "pdf_text": payload.get("pdf_text") or "",
        }
    )
    page_index = payload.get("page_index", 0)
    return {
        "job_id": payload.get("job_id"),
        "extract_mode": mode,
        "page_count": 1,
        "pages": [{"page_index": page_index, **extracted}],
    }
