"""Extract API — needs OCR from upload first; talks via job_id."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.extract_pipeline import process_job

router = APIRouter(prefix="/api", tags=["extract"])


@router.get("/extract/{job_id}")
async def extract_invoice_by_path(job_id: str):
    """
    Does not accept a PDF. Call upload first:

      1) POST /api/upload  →  { job_id, pages: [...ocr] }
      2) GET  /api/extract/{job_id}
    """
    try:
        return process_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
