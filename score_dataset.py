#!/usr/bin/env python3
"""OCR each Dataset PDF once, score rules vs groq against hand-checked truth."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"

# Ground truth from native PDF text (hand-checked).
TRUTH: dict[str, dict] = {
    "6934.pdf": {
        "seller_name": "Balaji Malts Private Limited",
        "seller_gstin": "24AACCB2151L1ZY",
        "invoice_no": "GJ/TR/006934",
        "invoice_date": "06/Jul/2025",
        "vehicle_no": "GJ12BV2970",
        "description": "Steam Coal",
        "hsn": "27011920",
        "qty": 40.61,
        "rate": 4600.0,
        "amount": 186806.0,
        "cgst": 4670.15,
        "sgst": 4670.15,
        "cess": 16244.0,
        "tcs": 2123.9,
        "total_amount": 214514.0,
    },
    "GJ03BV6630.pdf": {
        "seller_name": "Agarwal Coal Corporation Pvt. Ltd.",
        "seller_gstin": "24AACCA8468K1ZD",
        "invoice_no": "NV1125011666",
        "invoice_date": "05-08-2025",
        "vehicle_no": "GJ03BV6630",
        "description": "Imported Coal",
        "hsn": "270119",
        "qty": 43.51,
        "rate": 5640.0,
        "amount": 245396.4,
        "cgst": 6134.91,
        "sgst": 6134.91,
        "cess": 17404.0,
        "tcs": 2750.7,
        "total_amount": 277820.92,
    },
    "INVOICE3715.pdf": {
        "seller_name": "Green Gold Global Resources Pvt. Ltd.",
        "seller_gstin": "24AAICG5558N1Z2",
        "invoice_no": "GGLSUK022305172",
        "invoice_date": "08-Feb-2024",
        "vehicle_no": "GJ18AZ3715",
        "description": "Imported Coal",
        "hsn": "27011920",
        "qty": 43.91,
        "rate": 8000.0,
        "amount": 351280.0,
        "cgst": 8782.0,
        "sgst": 8782.0,
        "cess": 17564.0,
        "tcs": 3865.0,
        "total_amount": 390273.0,
    },
    "Invoice_VGIL2526ND1798.pdf": {
        "seller_name": "Vasuki Global Industries Limited",
        "seller_gstin": "24AAFCV6687A1Z9",
        "invoice_no": "VGIL2526ND1798",
        "invoice_date": "03/05/2025",
        "vehicle_no": "GJ03BV6580",
        "description": "Indonesian Steam Coal",
        "hsn": "27011920",
        "qty": 41.14,
        "rate": 4750.0,
        "amount": 195415.0,
        "cgst": 4885.38,
        "sgst": 4885.38,
        "cess": 16456.0,
        "tcs": 2216.42,
        "total_amount": 223858.0,
    },
    "SalesBill_GT_5776.PDF": {
        "seller_name": "Angel Resources Private Limited",
        "seller_gstin": "24ABBCA6392E1ZU",
        "invoice_no": "GT/5776",
        "invoice_date": "13/08/2025",
        "vehicle_no": "GJ12BV1475",
        "description": "Imported Coal",
        "hsn": "27011920",
        "qty": 43.77,
        "rate": 4375.0,
        "amount": 191493.75,
        "cgst": 4787.34,
        "sgst": 4787.34,
        "cess": 17508.0,
        "tcs": 2185.76,
        "total_amount": 220762.0,
    },
    "Sales_2775 SANMAT.pdf": {
        "seller_name": "Honestfalcon Resources Pvt Ltd",
        "seller_gstin": "24AAGCH8484E1ZF",
        "invoice_no": "2775",
        "invoice_date": "1-Jun-25",
        "vehicle_no": "GJ03BZ6663",
        "description": "Imported Coal",
        "hsn": "27011920",
        "qty": 44.8,
        "rate": 4500.0,
        "amount": 201600.0,
        "cgst": 5040.0,
        "sgst": 5040.0,
        "cess": 17920.0,
        "tcs": 2296.0,
        "total_amount": 231896.0,
    },
}

FIELDS = list(next(iter(TRUTH.values())).keys())


def _norm_str(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\bprivate limited\b", "pvt ltd", s)
    s = re.sub(r"\bpvt\.?\s*ltd\.?\b", "pvt ltd", s)
    s = re.sub(r"\blimited\b", "ltd", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _match(key: str, truth, got) -> bool:
    if truth is None and got is None:
        return True
    if got is None:
        return False
    if key in {
        "qty",
        "rate",
        "amount",
        "cgst",
        "sgst",
        "cess",
        "tcs",
        "total_amount",
    }:
        try:
            return abs(float(got) - float(truth)) < 0.05
        except (TypeError, ValueError):
            return False
    if key == "hsn":
        return str(got).lstrip("0") == str(truth).lstrip("0") or str(truth) in str(got) or str(got) in str(
            truth
        )
    if key == "description":
        a, b = _norm_str(str(truth)), _norm_str(str(got))
        return a in b or b in a or a[:12] == b[:12]
    if key == "seller_name":
        a, b = _norm_str(str(truth)), _norm_str(str(got))
        return a in b or b in a or a[:16] == b[:16]
    if key == "invoice_date":
        # ignore separator/format noise: keep digits+month letters
        def dig(x: str) -> str:
            return re.sub(r"[^0-9A-Za-z]", "", str(x)).lower()

        return dig(str(truth)) == dig(str(got)) or dig(str(truth)) in dig(str(got))
    if key == "vehicle_no":
        return re.sub(r"\s+", "", str(got).upper()) == re.sub(r"\s+", "", str(truth).upper())
    if key == "seller_gstin":
        return str(got).upper() == str(truth).upper()
    if key == "invoice_no":
        return re.sub(r"\s+", "", str(got).upper()) == re.sub(r"\s+", "", str(truth).upper())
    return _norm_str(str(truth)) == _norm_str(str(got))


def score(fields: dict, truth: dict) -> tuple[int, list[str]]:
    ok = 0
    misses: list[str] = []
    for k in FIELDS:
        if _match(k, truth[k], fields.get(k)):
            ok += 1
        else:
            misses.append(f"{k}: truth={truth[k]!r} got={fields.get(k)!r}")
    return ok, misses


def main() -> None:
    load_dotenv(ROOT / ".env")
    from services.extract_pipeline import extract_pages, extract_mode
    from services.upload_pipeline import process_pdf_bytes, safe_stem

    names = sys.argv[1:] or sorted(TRUTH.keys())
    rows = []

    for name in names:
        if name not in TRUTH:
            print(f"skip unknown {name}")
            continue
        path = DATASET / name
        if not path.is_file():
            print(f"missing {path}")
            continue
        print(f"\n### OCR {name}")
        handoff = process_pdf_bytes(path.read_bytes(), stem=safe_stem(name))
        pages = handoff["pages"]
        truth = TRUTH[name]

        os.environ["EXTRACT_MODE"] = "rules"
        rules = extract_pages(pages, job_dir=None, handoff=handoff)[0]["fields"]
        r_ok, r_miss = score(rules, truth)

        os.environ["EXTRACT_MODE"] = "groq"
        # extract_mode reads env each call
        assert extract_mode() == "groq"
        groq = extract_pages(pages, job_dir=None, handoff=handoff)[0]["fields"]
        g_ok, g_miss = score(groq, truth)

        n = len(FIELDS)
        print(f"rules {r_ok}/{n}")
        if r_miss:
            for m in r_miss:
                print(f"  - {m}")
        print(f"groq  {g_ok}/{n}")
        if g_miss:
            for m in g_miss:
                print(f"  - {m}")
        rows.append((name, r_ok, g_ok, n, handoff["job_id"]))

    print("\n" + "=" * 72)
    print(f"{'PDF':<32} {'rules':>8} {'groq':>8}")
    print("-" * 72)
    tr = tg = 0
    for name, r_ok, g_ok, n, _ in rows:
        tr += r_ok
        tg += g_ok
        print(f"{name:<32} {r_ok:>3}/{n:<3}  {g_ok:>3}/{n:<3}")
    total = len(rows) * len(FIELDS)
    print("-" * 72)
    print(f"{'TOTAL':<32} {tr:>3}/{total:<3}  {tg:>3}/{total:<3}")
    if total:
        print(f"{'SCORE %':<32} {100*tr/total:6.1f}%  {100*tg/total:6.1f}%")

    out = ROOT / "uploads" / "score_summary.json"
    out.write_text(
        json.dumps(
            {
                "per_pdf": [
                    {"pdf": n, "rules": r, "groq": g, "fields": f, "job_id": j}
                    for n, r, g, f, j in rows
                ],
                "totals": {"rules": tr, "groq": tg, "possible": total},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
