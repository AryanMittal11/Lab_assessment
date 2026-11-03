#!/usr/bin/env python3
"""
Run OCR + extract on one Dataset PDF (no server needed).

Usage:
  python run_pdf.py                  # uses PDF below
  python run_pdf.py 6934.pdf
  python run_pdf.py Invoice_VGIL2526ND1798.pdf
  python run_pdf.py /full/path/to/file.pdf
  EXTRACT_MODE=groq python run_pdf.py 6934.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"

# Change this (or pass a CLI arg) to pick which PDF to run.
DEFAULT_PDF = "6934.pdf"


def resolve_pdf(arg: str | None) -> Path:
    name = (arg or DEFAULT_PDF).strip()
    path = Path(name).expanduser()
    if not path.is_absolute():
        # Prefer Dataset/<name>, then project-relative path
        candidates = [DATASET / path.name, ROOT / path, path]
        path = next((p for p in candidates if p.is_file()), DATASET / path.name)
    if not path.is_file():
        available = sorted(p.name for p in DATASET.iterdir() if p.suffix.lower() == ".pdf")
        raise SystemExit(
            f"PDF not found: {name}\n"
            f"Dataset PDFs: {', '.join(available) or '(none)'}"
        )
    return path.resolve()


def main() -> None:
    load_dotenv(ROOT / ".env")

    # Import after dotenv so EXTRACT_MODE / GROQ_* are visible
    from services.extract_pipeline import extract_mode, process_job
    from services.upload_pipeline import process_pdf_bytes, safe_stem

    pdf_path = resolve_pdf(sys.argv[1] if len(sys.argv) > 1 else None)
    mode = extract_mode()
    print(f"PDF:          {pdf_path}")
    print(f"EXTRACT_MODE: {mode}")
    print("Running OCR…")

    handoff = process_pdf_bytes(pdf_path.read_bytes(), stem=safe_stem(pdf_path.name))
    job_id = handoff["job_id"]
    print(f"job_id:       {job_id}")
    print(f"pages:        {handoff['page_count']}")
    print(f"OCR handoff:  {handoff['upload_dir']}/ocr_handoff.json")
    print(f"Extracting ({mode})…")

    result = process_job(job_id)
    out_path = Path(result["upload_dir"]) / "extract_handoff.json"
    print(f"Extract out:  {out_path}")
    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
