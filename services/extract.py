"""Step 4 — rule-based field extraction from one page of OCR handoff (no LLM)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

GSTIN_RE = re.compile(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9])\b", re.I)

VEHICLE_RE = re.compile(r"\b([A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{3,4})\b", re.I)

DATE_RE = re.compile(
    r"\b(\d{1,2}[/.\-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})"
    r"[/.\-]\d{2,4})\b",
    re.I,
)

MONEY_RE = re.compile(r"\b(\d{1,3}(?:,\d{2,3})*\.\d{2}|\d+\.\d{2})\b")
QTY_RE = re.compile(r"\b(\d+\.\d{3})\b")  # e.g. 40.610 MT

# Longer / more specific aliases first within each field.
LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "invoice_no": (
        "invoice number",
        "invoice no.",
        "invoice no",
        "inv. no.",
        "inv. no",
        "inv no",
    ),
    "invoice_date": (
        "invoice date",
        "inv date",
        "bill date",
        "ack. date",
        "ack date",
    ),
    "vehicle_no": (
        "truck no.",
        "truck no",
        "vehicle no.",
        "vehicle no",
        "veh no",
        "lorry no",
    ),
    "eway_bill": (
        "e-way bill no.",
        "e-way bill no",
        "eway bill no",
        "e way bill no",
    ),
    "hsn": ("hsn code", "hsn/sac", "hsn"),
    "cgst": ("cgst",),
    "sgst": ("sgst",),
    "igst": ("igst",),
    "cess": ("cess",),
    "tcs": ("tcs",),
    "total_amount": (
        "grand total",
        "invoice total",
        "total amount",
        "net amount",
        "total payable",
    ),
    "seller_gstin": ("gstin no.", "gstin no", "gstin", "gst no"),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _parse_amount(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = re.sub(r"(?i)^rs\.?\s*", "", str(raw).replace(",", "").strip())
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _row_texts(row: list[dict]) -> list[str]:
    return [_normalize(b.get("text", "")) for b in row if b.get("text")]


def _joined_row(row: list[dict]) -> str:
    return " ".join(_row_texts(row))


def _label_pattern(alias: str) -> re.Pattern[str]:
    """Word-boundary label; optional trailing punctuation / separator."""
    esc = re.escape(alias)
    return re.compile(rf"(?i)(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])\s*[:.\-]?\s*")


def _value_after_alias(text: str, alias: str) -> str | None:
    m = _label_pattern(alias).search(text)
    if not m:
        return None
    rest = _normalize(text[m.end() :])
    return rest or None


def _find_labeled_values(layout_rows: list[list[dict]]) -> dict[str, str]:
    """Pull values that sit on the same row as a known label."""
    found: dict[str, str] = {}

    for row in layout_rows:
        texts = _row_texts(row)
        if not texts:
            continue
        joined = " ".join(texts)

        for field, aliases in LABEL_ALIASES.items():
            if field in found:
                continue
            for alias in aliases:
                # 1) Label + value in the joined row string
                val = _value_after_alias(joined, alias)
                if val:
                    # stop at next obvious label boundary for noisy OCR rows
                    val = re.split(
                        r"\s(?=(?:GSTIN|State|Pin|E-Way|Ack\.|CGST|SGST|Taxable)\b)",
                        val,
                        maxsplit=1,
                        flags=re.I,
                    )[0].strip(" :.-")
                    if val:
                        found[field] = val
                        break

                # 2) Label in one cell, value in a later cell (often ": GJ12...")
                for i, t in enumerate(texts):
                    if not _label_pattern(alias).search(t):
                        continue
                    inline = _value_after_alias(t, alias)
                    if inline and not _looks_like_bare_label(inline):
                        found[field] = inline.lstrip(": ").strip()
                        break
                    for j in range(i + 1, min(i + 4, len(texts))):
                        cand = texts[j].lstrip(" :.-")
                        if not cand or _looks_like_bare_label(cand):
                            continue
                        found[field] = cand
                        break
                    if field in found:
                        break
            if field in found:
                break

    return found


def _looks_like_bare_label(text: str) -> bool:
    t = text.lower().rstrip(" .:/-")
    for aliases in LABEL_ALIASES.values():
        if t in aliases:
            return True
    return bool(
        re.fullmatch(
            r"(transporter|truck no\.?|payment terms|taxable value|unit|total|date|rate|amount.*)",
            t,
            re.I,
        )
    )


def _first_gstin(text: str) -> str | None:
    m = GSTIN_RE.search(text.upper())
    return m.group(1).upper() if m else None


def _first_vehicle(text: str) -> str | None:
    compact = re.sub(r"\s+", "", text.upper())
    for m in VEHICLE_RE.finditer(compact):
        raw = re.sub(r"\s+", "", m.group(1)).upper()
        digits = sum(c.isdigit() for c in raw)
        if digits < 4 or GSTIN_RE.fullmatch(raw):
            continue
        # typical Indian plates end with 3–4 digits
        if re.search(r"\d{3,4}$", raw):
            return raw
    return None


_COMPANY_RE = re.compile(
    r"([A-Z][A-Za-z0-9&.,'()\- ]{3,}(?:Private Limited|Pvt\.?\s*Ltd\.?|Ltd\.?|LLP))",
    re.I,
)

_BUYER_SPLIT_RE = re.compile(
    r"(?i)Details of Receiver|Billed To|Shipped to Party|Place of Supply",
)


def _issuer_header(text: str) -> str:
    """Text before buyer/receiver block — usually the seller header."""
    if not text:
        return ""
    return _BUYER_SPLIT_RE.split(text, maxsplit=1)[0]


def _dedupe_company(name: str) -> str:
    name = _normalize(name)
    parts = name.split()
    half = len(parts) // 2
    if half and parts[:half] == parts[half:]:
        return " ".join(parts[:half])
    return name


def _guess_seller_name(
    layout_rows: list[list[dict]],
    full_text: str,
    pdf_text: str = "",
) -> str | None:
    # Prefer native PDF header (OCR often drops issuer, keeps buyer).
    for source in (pdf_text, full_text):
        header = _issuer_header(source)
        m = _COMPANY_RE.search(header)
        if m:
            return _dedupe_company(m.group(1))

    for row in layout_rows:
        joined = _joined_row(row)
        if _BUYER_SPLIT_RE.search(joined):
            continue
        m = _COMPANY_RE.search(joined)
        if not m:
            continue
        return _dedupe_company(m.group(1))

    m = _COMPANY_RE.search(full_text)
    return _dedupe_company(m.group(1)) if m else None


def _pick_seller_gstin(full_text: str, labeled: dict[str, str], pdf_text: str = "") -> str | None:
    for source in (pdf_text, full_text):
        header = _issuer_header(source)
        gstin = _first_gstin(header)
        if gstin:
            return gstin
    if labeled.get("seller_gstin"):
        gstin = _first_gstin(labeled["seller_gstin"])
        if gstin:
            return gstin
    return _first_gstin(pdf_text) or _first_gstin(full_text)


def _guess_description(layout_rows: list[list[dict]]) -> str | None:
    goods = re.compile(
        r"\b(steam\s+coal|coal|coke|iron\s+ore|cement|diesel|petrol|sand|steel)\b",
        re.I,
    )
    skip = re.compile(r"description|receiver|billed|supply|gst|total|amount|declaration", re.I)
    for row in layout_rows:
        texts = _row_texts(row)
        if len(texts) == 1 and goods.search(texts[0]) and not skip.search(texts[0]):
            return texts[0]
    for row in layout_rows:
        for t in _row_texts(row):
            if goods.search(t) and not skip.search(t):
                m = goods.search(t)
                return _normalize(m.group(0)).title() if m else t
    return None


def _pick_invoice_no(labeled: dict[str, str], full_text: str) -> str | None:
    # Strong pattern used on these GST coal invoices
    m = re.search(r"\b([A-Z]{1,6}/[A-Z0-9]+/\d{3,})\b", full_text, re.I)
    if m:
        return m.group(1).upper()

    raw = labeled.get("invoice_no")
    if raw:
        # take first token that has a digit
        for tok in re.split(r"\s+", raw):
            tok = tok.strip(" :,-")
            if re.search(r"\d", tok) and len(tok) >= 3:
                if _first_gstin(tok):
                    continue
                return tok

    m = re.search(r"(?i)invoice\s*no\.?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,})", full_text)
    if m:
        cand = m.group(1).strip()
        if not cand.lower().startswith("e-way"):
            return cand
    return None


def _pick_invoice_date(labeled: dict[str, str], full_text: str) -> str | None:
    for key in ("invoice_date",):
        if labeled.get(key):
            m = DATE_RE.search(labeled[key])
            if m:
                return m.group(1)

    for label in ("Invoice Date", "Bill Date", "Ack. Date", "Ack Date"):
        m = re.search(rf"{re.escape(label)}\s*[-:]?\s*({DATE_RE.pattern})", full_text, re.I)
        if m:
            return m.group(1)

    # Prefer a date that sits near invoice meta, else first date on page
    m = DATE_RE.search(full_text)
    return m.group(1) if m else None


def _pick_hsn(layout_rows: list[list[dict]], full_text: str) -> str | None:
    for row in layout_rows:
        texts = _row_texts(row)
        joined = " ".join(texts)
        if re.search(r"\bhsn\b", joined, re.I):
            m = re.search(r"\b(\d{4,8})\b", joined)
            if m:
                return m.group(1)
        for t in texts:
            if re.fullmatch(r"\d{6,8}", t):
                return t
    m = re.search(r"\b(2701\d{0,4}|\d{8})\b", full_text)
    return m.group(1) if m else None


def _money_values(full_text: str) -> list[float]:
    vals = []
    for m in MONEY_RE.finditer(full_text):
        v = _parse_amount(m.group(1))
        if v is not None:
            vals.append(v)
    return vals


def _tax_amounts(full_text: str, layout_rows: list[list[dict]]) -> dict[str, float | None]:
    """
    Tax line amounts — never the @rate percent.
    Prefer duplicated money values near the bottom (CGST/SGST often equal).
    """
    out: dict[str, float | None] = {"cgst": None, "sgst": None, "cess": None, "tcs": None}

    # Explicit "CGST ... 4670.15" within a window after label (skip @ 2.50%)
    for field in ("cgst", "sgst", "cess", "tcs"):
        window_size = 200 if field in ("cess", "tcs") else 80
        for alias in LABEL_ALIASES[field]:
            for m in re.finditer(rf"(?i)\b{re.escape(alias)}\b", full_text):
                window = full_text[m.end() : m.end() + window_size]
                # skip percentage then find money
                window_wo_pct = re.sub(r"@\s*[\d.]+%\s*", " ", window)
                nums = MONEY_RE.findall(window_wo_pct)
                # filter out tiny rates mistaken as money (2.50 already removed via %)
                parsed = [_parse_amount(n) for n in nums]
                parsed = [p for p in parsed if p is not None and p >= 10]
                if parsed:
                    out[field] = parsed[0]
                    break
            if out[field] is not None:
                break

    # Equal CGST/SGST heuristic from repeated money amounts in 100–50000 band
    if out["cgst"] is None or out["sgst"] is None:
        counts = Counter(_money_values(full_text))
        for val, cnt in counts.most_common():
            if cnt >= 2 and 50 <= val <= 50000:
                if out["cgst"] is None:
                    out["cgst"] = val
                if out["sgst"] is None:
                    out["sgst"] = val
                break

    # TCS often unique smaller amount near "TCS @ 1%"
    if out["tcs"] is None:
        m = re.search(r"(?i)\btcs\b.{0,40}", full_text)
        if m:
            # look in following ~300 chars of page for a plausible tcs amount
            pass
        # From sample: 2123.90 appears once near tax block
        for row in layout_rows[-8:]:
            for t in _row_texts(row):
                for n in MONEY_RE.findall(t):
                    v = _parse_amount(n)
                    if v is None:
                        continue
                    if out["cgst"] and abs(v - out["cgst"]) < 0.01:
                        continue
                    if 20 <= v <= 20000 and out["tcs"] is None:
                        # defer — set later from unique candidates
                        pass

    return out


def _pick_qty(full_text: str, layout_rows: list[list[dict]]) -> float | None:
    # Prefer number next to MT / M.T / Quintal etc.
    m = re.search(
        r"(\d+\.\d{3}|\d+\.\d{1,2}|\d+)\s*(?:MT|M\.T\.?|MTS|QTL|Quintal|Kg|Kgs)\b",
        full_text,
        re.I,
    )
    if m:
        return _parse_amount(m.group(1))

    # Row that is only a qty-like value (40.610), ignore "@ 1.000%" style
    candidates: list[float] = []
    for row in layout_rows:
        joined = _joined_row(row)
        if "@" in joined or "%" in joined:
            continue
        for t in _row_texts(row):
            if QTY_RE.fullmatch(t):
                v = float(t)
                if v >= 1:
                    candidates.append(v)
    if candidates:
        # coal loads are usually >> 1
        return max(candidates)
    return None


def _line_numbers(full_text: str, layout_rows: list[list[dict]]) -> dict[str, float | None]:
    money = _money_values(full_text)
    qty = _pick_qty(full_text, layout_rows)

    # Rate: prefer values near "Rate" label or typical mid range (not pin codes)
    rate = None
    for row in layout_rows:
        joined = _joined_row(row)
        if re.search(r"(?i)(?<![A-Za-z])rate(?![A-Za-z])", joined):
            nums = [_parse_amount(n) for n in MONEY_RE.findall(joined)]
            nums = [n for n in nums if n is not None and 10 <= n <= 100000]
            if nums:
                rate = nums[0]
                break
    if rate is None:
        # standalone row like "4600.00 : To PIN..."
        for row in layout_rows:
            texts = _row_texts(row)
            for t in texts:
                m = re.match(r"^([\d,]+\.\d{2})\b", t)
                if not m:
                    continue
                v = _parse_amount(m.group(1))
                if v is not None and 100 <= v <= 50000:
                    rate = v
                    break
            if rate is not None:
                break

    # Line / taxable amount and grand total: largest money figures
    # Exclude pure tax duplicates if possible
    significant = sorted({m for m in money if m >= 1000})
    amount = None
    total = None
    if len(significant) >= 2:
        total = significant[-1]
        amount = significant[-2]
        # if top two are equal (OCR dup), step down
        if abs(total - amount) < 0.01 and len(significant) >= 3:
            amount = significant[-3]
    elif len(significant) == 1:
        amount = total = significant[0]

    # Prefer "Total GST Payable" only when nothing larger exists
    m = re.search(r"(?i)total\s+gst\s+payable\s*:\s*rs\.?\s*([\d,]+\.?\d*)", full_text)
    gst_payable = _parse_amount(m.group(1)) if m else None
    if total is None and gst_payable is not None:
        total = gst_payable
    if total is not None and gst_payable is not None and abs(total - gst_payable) < 0.01:
        # total wrongly set to GST payable — bump to max money if larger exists
        bigger = [x for x in money if x > gst_payable * 2]
        if bigger:
            total = max(bigger)
            others = sorted({x for x in money if x >= 1000 and abs(x - total) > 0.01})
            if others:
                amount = others[-1]

    return {"qty": qty, "rate": rate, "amount": amount, "total_amount": total}


def extract_from_ocr(ocr_out: dict[str, Any]) -> dict[str, Any]:
    """
    Consume one page OCR dict; return fixed invoice fields (null if missing).

    Input keys: text_blocks, full_text, layout_rows, tables (optional),
    pdf_text (optional native PDF text — preferred for seller header).
    """
    full_text = ocr_out.get("full_text") or ""
    pdf_text = ocr_out.get("pdf_text") or ""
    layout_rows: list[list[dict]] = ocr_out.get("layout_rows") or []

    labeled = _find_labeled_values(layout_rows)
    nums = _line_numbers(full_text, layout_rows)
    taxes = _tax_amounts(full_text, layout_rows)

    vehicle = None
    if labeled.get("vehicle_no"):
        vehicle = _first_vehicle(labeled["vehicle_no"]) or labeled["vehicle_no"].lstrip(": ").split()[0]
    vehicle = _first_vehicle(full_text) or vehicle

    # Fill remaining tax amounts from unique money near the tax footer.
    # On these invoices: CESS is often a mid unique figure; TCS a smaller one.
    skip = {
        taxes.get("cgst"),
        taxes.get("sgst"),
        nums.get("amount"),
        nums.get("total_amount"),
        nums.get("rate"),
    }
    skip = {s for s in skip if s is not None}
    money_src = "\n".join(x for x in (pdf_text, full_text) if x)
    counts = Counter(_money_values(money_src))
    uniques = sorted(
        (v for v, c in counts.items() if c == 1 and 20 <= v <= 50000 and v not in skip),
        reverse=True,
    )
    has_cess_label = bool(re.search(r"(?i)\bcess\b", money_src))
    has_tcs_label = bool(re.search(r"(?i)\btcs\b", money_src))

    if taxes["cess"] is None and has_cess_label and uniques:
        # Prefer larger leftover (CESS) when both CESS + TCS exist
        if has_tcs_label and len(uniques) >= 2:
            taxes["cess"] = uniques[0]
            if taxes["tcs"] is None:
                taxes["tcs"] = uniques[1]
        elif not has_tcs_label:
            taxes["cess"] = uniques[0]

    if taxes["tcs"] is None and has_tcs_label:
        last_money = _money_values(money_src)[-8:]
        for val in last_money:
            if val in skip:
                continue
            if taxes.get("cgst") and abs(val - taxes["cgst"]) < 0.01:
                continue
            if taxes.get("cess") and abs(val - taxes["cess"]) < 0.01:
                continue
            if 20 <= val <= 30000:
                taxes["tcs"] = val
        if taxes["tcs"] is None:
            for val in reversed(uniques):
                if taxes.get("cess") and abs(val - taxes["cess"]) < 0.01:
                    continue
                taxes["tcs"] = val
                break

    # Prefer PDF text for totals when OCR scrambled the footer
    if pdf_text:
        pdf_nums = _line_numbers(pdf_text, [])
        if nums.get("total_amount") is None and pdf_nums.get("total_amount") is not None:
            nums["total_amount"] = pdf_nums["total_amount"]
        if nums.get("amount") is None and pdf_nums.get("amount") is not None:
            nums["amount"] = pdf_nums["amount"]
        if nums.get("rate") is None and pdf_nums.get("rate") is not None:
            nums["rate"] = pdf_nums["rate"]
        if nums.get("qty") is None and pdf_nums.get("qty") is not None:
            nums["qty"] = pdf_nums["qty"]
        pdf_taxes = _tax_amounts(pdf_text, [])
        for k in ("cgst", "sgst", "cess", "tcs"):
            if taxes.get(k) is None and pdf_taxes.get(k) is not None:
                taxes[k] = pdf_taxes[k]

    fields = {
        "seller_name": _guess_seller_name(layout_rows, full_text, pdf_text=pdf_text),
        "seller_gstin": _pick_seller_gstin(full_text, labeled, pdf_text=pdf_text),
        "invoice_no": _pick_invoice_no(labeled, full_text),
        "invoice_date": _pick_invoice_date(labeled, full_text),
        "vehicle_no": vehicle,
        "description": _guess_description(layout_rows),
        "hsn": _pick_hsn(layout_rows, full_text),
        "qty": nums.get("qty"),
        "rate": nums.get("rate"),
        "amount": nums.get("amount"),
        "cgst": taxes.get("cgst"),
        "sgst": taxes.get("sgst"),
        "cess": taxes.get("cess"),
        "tcs": taxes.get("tcs"),
        "total_amount": nums.get("total_amount"),
    }

    return {
        "fields": fields,
        "method": "rules_v1",
    }
