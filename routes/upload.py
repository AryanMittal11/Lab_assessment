"""Upload API — PDF in, OCR handoff JSON out (no extract)."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.upload_pipeline import process_pdf_bytes, safe_stem

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_invoice(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return process_pdf_bytes(data, stem=safe_stem(file.filename))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
