"""Step 3 — PaddleOCR 3.x: text, bounding boxes, tables (handoff for extraction)."""

from __future__ import annotations

from typing import Any

import numpy as np

_ocr = None
_structure = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR

        _ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    return _ocr


def _get_structure():
    global _structure
    if _structure is False:
        return None
    if _structure is None:
        try:
            from paddleocr import PPStructureV3

            _structure = PPStructureV3(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_table_recognition=True,
                use_seal_recognition=False,
                use_formula_recognition=False,
                use_chart_recognition=False,
            )
        except Exception:
            _structure = False
            return None
    return _structure


def _as_dict(result_item: Any) -> dict:
    if isinstance(result_item, dict):
        return result_item
    if hasattr(result_item, "json"):
        raw = result_item.json
        data = raw() if callable(raw) else raw
        if isinstance(data, dict):
            return data.get("res", data) if "res" in data and isinstance(data["res"], dict) else data
    if hasattr(result_item, "keys"):
        return dict(result_item)
    return {}


def _polygon_to_bbox(poly: list) -> dict[str, float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return {
        "x_min": min(xs),
        "y_min": min(ys),
        "x_max": max(xs),
        "y_max": max(ys),
    }


def _group_into_rows(blocks: list[dict], y_tol: float = 12.0) -> list[list[dict]]:
    if not blocks:
        return []

    sorted_blocks = sorted(
        blocks,
        key=lambda b: (b["bbox"]["y_min"] + b["bbox"]["y_max"]) / 2.0,
    )
    rows: list[list[dict]] = []
    current: list[dict] = [sorted_blocks[0]]
    current_y = (sorted_blocks[0]["bbox"]["y_min"] + sorted_blocks[0]["bbox"]["y_max"]) / 2.0

    for block in sorted_blocks[1:]:
        y = (block["bbox"]["y_min"] + block["bbox"]["y_max"]) / 2.0
        if abs(y - current_y) <= y_tol:
            current.append(block)
        else:
            current.sort(key=lambda b: b["bbox"]["x_min"])
            rows.append(current)
            current = [block]
            current_y = y

    current.sort(key=lambda b: b["bbox"]["x_min"])
    rows.append(current)
    return rows


def _parse_ocr_blocks(result_item: Any) -> list[dict[str, Any]]:
    data = _as_dict(result_item)

    texts = data.get("rec_texts") or data.get("texts") or []
    scores = data.get("rec_scores") or data.get("scores") or []
    polys = data.get("rec_polys") or data.get("dt_polys") or data.get("polys") or []

    if not texts and isinstance(result_item, list):
        blocks = []
        for line in result_item:
            if not line:
                continue
            try:
                quad, (text, conf) = line
            except (TypeError, ValueError):
                continue
            blocks.append(
                {
                    "text": text,
                    "confidence": float(conf),
                    "bbox": _polygon_to_bbox(quad),
                    "polygon": [[float(x), float(y)] for x, y in quad],
                }
            )
        return blocks

    blocks: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        poly = polys[i] if i < len(polys) else None
        conf = float(scores[i]) if i < len(scores) else 0.0
        if poly is None:
            continue
        poly_list = [[float(x), float(y)] for x, y in poly]
        blocks.append(
            {
                "text": str(text),
                "confidence": conf,
                "bbox": _polygon_to_bbox(poly_list),
                "polygon": poly_list,
            }
        )
    return blocks


def _run_pp_structure(image: np.ndarray) -> list[dict[str, Any]]:
    engine = _get_structure()
    if engine is None:
        return []

    try:
        raw = list(engine.predict(image))
    except Exception:
        return []

    tables: list[dict[str, Any]] = []
    for item in raw:
        data = _as_dict(item)
        table_html = data.get("table_html") or data.get("html")
        table_list = data.get("tables") or data.get("table_res_list") or []

        if table_html:
            tables.append({"html": table_html, "bbox": None, "cells": None})

        for t in table_list:
            if not isinstance(t, dict):
                continue
            bbox = t.get("bbox") or t.get("coordinate")
            tables.append(
                {
                    "html": t.get("html") or t.get("pred_html"),
                    "bbox": (
                        {
                            "x_min": float(bbox[0]),
                            "y_min": float(bbox[1]),
                            "x_max": float(bbox[2]),
                            "y_max": float(bbox[3]),
                        }
                        if bbox is not None and len(bbox) >= 4
                        else None
                    ),
                    "cells": t.get("cell_bbox") or t.get("cells"),
                }
            )
    return tables


def run_ocr(image: np.ndarray) -> dict[str, Any]:
    """OCR + layout handoff payload for the extraction developer."""
    ocr = _get_ocr()
    raw = list(ocr.predict(image))
    result_item = raw[0] if raw else {}

    text_blocks = _parse_ocr_blocks(result_item)
    layout_rows = _group_into_rows(text_blocks)
    full_text = "\n".join(" ".join(b["text"] for b in row) for row in layout_rows)
    tables = _run_pp_structure(image)

    return {
        "text_blocks": text_blocks,
        "full_text": full_text,
        "layout_rows": [
            [{"text": b["text"], "confidence": b["confidence"], "bbox": b["bbox"]} for b in row]
            for row in layout_rows
        ],
        "tables": tables,
    }
