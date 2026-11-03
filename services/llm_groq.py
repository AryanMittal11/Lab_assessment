"""Groq LLM invoice field extraction — same schema as rules_v1."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from groq import BadRequestError, Groq

FIELD_KEYS = (
    "seller_name",
    "seller_gstin",
    "invoice_no",
    "invoice_date",
    "vehicle_no",
    "description",
    "hsn",
    "qty",
    "rate",
    "amount",
    "cgst",
    "sgst",
    "cess",
    "tcs",
    "total_amount",
)

_NUMERIC_KEYS = frozenset(
    {"qty", "rate", "amount", "cgst", "sgst", "cess", "tcs", "total_amount"}
)

_SYSTEM = """You extract fields from Indian GST tax invoices.
Return ONLY a JSON object with exactly these keys:
seller_name, seller_gstin, invoice_no, invoice_date, vehicle_no, description,
hsn, qty, rate, amount, cgst, sgst, cess, tcs, total_amount.

Rules:
- Use null when a value is missing or unclear. Never invent values.
- qty, rate, amount, cgst, sgst, cess, tcs, total_amount must be plain numbers (no Rs/commas) or null.
- seller_name / seller_gstin: the ISSUER at the top (company that prints TAX INVOICE). NEVER the receiver / billed-to / shipped-to party.
- invoice_no: tax invoice number (e.g. GJ/TR/006934). Not Ack No, IRN, e-way bill, L.R No, or verbal order refs.
- invoice_date as printed (e.g. 06/Jul/2025). Prefer invoice/bill date over order date.
- vehicle_no: Indian truck/vehicle registration only (e.g. GJ12BV2970). NEVER Job Order No / L.R No / invoice refs like GJ/0232.
- qty: goods quantity (often ###.### with unit MT), NEVER S. No. / line index (1, 2, …).
- amount = taxable value; total_amount = grand/invoice total (words like "Two Lacs..." match this). Include all taxes — do NOT use Total GST Payable alone.
- rate = unit rate (e.g. 4600), not totals.
- Prefer PDF TEXT (native) over OCR when both are present.
"""


def _compact_layout(ocr: dict[str, Any], max_rows: int = 80) -> str:
    rows = ocr.get("layout_rows") or []
    lines: list[str] = []
    for row in rows[:max_rows]:
        if not isinstance(row, list):
            continue
        parts = []
        for block in row:
            if isinstance(block, dict):
                t = str(block.get("text") or "").strip()
                if t:
                    parts.append(t)
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def _client() -> Groq:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=key)


def _parse_number(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = re.sub(r"(?i)^rs\.?\s*", "", str(raw).replace(",", "").strip())
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _normalize_fields(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in FIELD_KEYS:
        val = raw.get(key)
        if key in _NUMERIC_KEYS:
            out[key] = _parse_number(val)
        else:
            if val is None:
                out[key] = None
            else:
                s = str(val).strip()
                out[key] = s or None
    return out


def _extract_json(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        raise ValueError("Empty Groq response")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("Groq response is not JSON")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("Groq JSON root must be an object")
    return data


def _looks_like_vehicle(val: str | None) -> bool:
    if not val:
        return False
    compact = re.sub(r"\s+", "", val.upper())
    # Job/invoice refs like GJ/0232 are not plates
    if "/" in compact:
        return False
    return bool(re.fullmatch(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}", compact))


def _post_correct(fields: dict[str, Any], ocr: dict[str, Any]) -> dict[str, Any]:
    """Regex overrides for fields LLMs routinely mix up on columnar PDFs."""
    from services.extract import _first_vehicle, _pick_qty

    pdf_text = (ocr.get("pdf_text") or "").strip()
    full_text = (ocr.get("full_text") or "").strip()
    layout_rows = ocr.get("layout_rows") or []
    combined = "\n".join(x for x in (pdf_text, full_text) if x)

    vehicle = fields.get("vehicle_no")
    if not _looks_like_vehicle(vehicle if isinstance(vehicle, str) else None):
        plate = _first_vehicle(combined)
        if plate:
            fields["vehicle_no"] = plate

    qty = fields.get("qty")
    qty_from_text = _pick_qty(combined, layout_rows) or _pick_qty(pdf_text, []) or _pick_qty(
        full_text, layout_rows
    )
    # Replace missing/tiny qty (S.No. confusion) when text has a real load qty
    if qty_from_text is not None and (qty is None or qty < 2 or abs(qty - 1.0) < 1e-9):
        fields["qty"] = qty_from_text
    elif qty_from_text is not None and qty is not None and qty_from_text >= 2 and qty < 2:
        fields["qty"] = qty_from_text

    return fields


def extract_from_ocr(ocr: dict[str, Any]) -> dict[str, Any]:
    """
    Extract invoice fields via Groq from one page OCR dict.
    Expects keys: full_text and/or pdf_text, optionally layout_rows.
    Returns {"fields": {...}, "method": "groq_v1"}.
    """
    full_text = (ocr.get("full_text") or "").strip()
    pdf_text = (ocr.get("pdf_text") or "").strip()
    if not full_text and not pdf_text:
        raise ValueError("OCR full_text/pdf_text empty; cannot run Groq extract")

    max_chars = int(os.getenv("GROQ_OCR_MAX_CHARS", "12000"))
    prompt_full = full_text
    prompt_pdf = pdf_text
    if len(prompt_full) > max_chars:
        prompt_full = prompt_full[:max_chars] + "\n...[truncated]"
    if len(prompt_pdf) > max_chars:
        prompt_pdf = prompt_pdf[:max_chars] + "\n...[truncated]"

    layout = _compact_layout(ocr)
    if len(layout) > max_chars:
        layout = layout[:max_chars] + "\n...[truncated]"

    model = (os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b").strip()
    parts = ["Extract invoice fields.\n"]
    if prompt_pdf:
        parts.append(f"--- PDF TEXT (preferred) ---\n{prompt_pdf}\n--- END PDF TEXT ---\n")
    if layout:
        parts.append(f"--- OCR LAYOUT ROWS ---\n{layout}\n--- END LAYOUT ---\n")
    if prompt_full:
        parts.append(f"--- OCR FULL TEXT ---\n{prompt_full}\n--- END OCR TEXT ---")
    user = "\n".join(parts)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
    client = _client()
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except BadRequestError:
        # Some Groq models reject json_object; fall back to plain completion.
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
        )
    content = completion.choices[0].message.content or ""
    fields = _post_correct(_normalize_fields(_extract_json(content)), ocr)
    return {
        "fields": fields,
        "method": "groq_v1",
    }
