"""MOSS-Image Layout JSON validator / compiler (in-process wrapper).

The official `moss-image/skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`
is a CLI tool. To avoid spawning a subprocess from inside a ComfyUI node, we
inline its pure logic here. The math is identical:
  - bbox_coord_space = "abs", bbox_format = "xyxy" (drafts use absolute pixels
    with the canvas top-left as origin)
  - "abs_centralized" + "yxyx" are the model format: subtract canvas center and
    yxyx instead of xyxy
  - fractional edges are floored (top/left) and ceiled (right/bottom) before
    subtracting the center, matching the reference compiler

Validation rules mirror the upstream `compile_prompt.py`:
  - caption drafts: {"caption": "<nonempty>"}            → returns prose
  - layout drafts: width/height positive multiples of 16; bbox_coord_space
    == "abs"; bbox_format == "xyxy"; summary.caption nonempty; items[]
    flat with label ∈ {object, character, text} and 4-number bbox; text
    items require a literal `text` field

The `compile()` function returns either:
  - (prose_string, "", 0, 0)                                  for caption drafts
  - (compiled_json_string, compiled_json_string, w, h)        for layout drafts

Errors raise ValueError with a human-readable message; the calling node is
responsible for catching them and falling back to a plain-prompt path.
"""
from __future__ import annotations

import json
import math
from typing import Tuple


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _fields(value, allowed, required, location: str) -> None:
    _require(isinstance(value, dict), f"{location} must be an object")
    unknown = set(value.keys()) - allowed
    missing = required - set(value.keys())
    _require(not unknown, f"{location}: unknown fields {sorted(unknown)}")
    _require(not missing, f"{location}: missing fields {sorted(missing)}")


def _text(value, location: str) -> None:
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{location} must be nonempty text",
    )


def _strings(value, location: str) -> None:
    _require(isinstance(value, list), f"{location} must be an array")
    for idx, item in enumerate(value):
        _text(item, f"{location}[{idx}]")


def _centered_box(box, width: int, height: int, location: str):
    _require(
        isinstance(box, list) and len(box) == 4,
        f"{location} must be [x1,y1,x2,y2]",
    )
    _require(
        all(isinstance(v, (int, float)) and math.isfinite(v) for v in box),
        f"{location} must contain four finite numbers",
    )
    x1, y1, x2, y2 = box
    _require(x1 < x2 and y1 < y2, f"{location} must have positive width and height")
    _require(
        0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
        f"{location} exceeds the {width}x{height} canvas",
    )
    result = [
        math.floor(round(float(y1), 9)) - height // 2,
        math.floor(round(float(x1), 9)) - width // 2,
        math.ceil(round(float(y2), 9)) - height // 2,
        math.ceil(round(float(x2), 9)) - width // 2,
    ]
    _require(
        result[0] < result[2] and result[1] < result[3],
        f"{location} is too small at export precision",
    )
    return result


def compile_layout(draft: dict) -> Tuple[str, str, int, int]:
    """Validate a MOSS-Image draft and return the compiled model prompt.

    Returns (positive_prompt, layout_json, width, height):
      - Caption draft → (prose, "", 0, 0)
      - Layout draft  → (compact_json_line, compact_json_line, width, height)
    """
    _require(isinstance(draft, dict), "Draft must be a JSON object")

    if "caption" in draft:
        _fields(draft, {"caption"}, {"caption"}, "Caption draft")
        _text(draft["caption"], "caption")
        return draft["caption"], "", 0, 0

    root_fields = {"width", "height", "bbox_coord_space", "bbox_format", "summary", "items"}
    _fields(draft, root_fields, root_fields, "Layout draft")
    width, height = draft["width"], draft["height"]
    for name, value in (("width", width), ("height", height)):
        _require(
            type(value) is int and value > 0 and value % 16 == 0,
            f"{name} must be a positive integer multiple of 16",
        )
    _require(draft["bbox_coord_space"] == "abs", "Draft bbox_coord_space must be abs")
    _require(draft["bbox_format"] == "xyxy", "Draft bbox_format must be xyxy")

    summary = draft["summary"]
    _fields(summary, {"caption", "meta", "canvas", "palette"}, {"caption"}, "summary")
    _text(summary["caption"], "summary.caption")
    if "canvas" in summary:
        _require(
            summary["canvas"] in ("single", "multi_panel", "collage"),
            "summary.canvas must be single, multi_panel or collage",
        )
    for key in ("meta", "palette"):
        if key in summary:
            _strings(summary[key], f"summary.{key}")

    items = draft["items"]
    _require(
        isinstance(items, list) and bool(items),
        "items must be a nonempty flat array",
    )
    compiled_items = []
    for idx, item in enumerate(items):
        loc = f"items[{idx}]"
        _fields(
            item,
            {"label", "caption", "bbox", "text", "tags", "palette"},
            {"label", "caption", "bbox"},
            loc,
        )
        _require(
            item["label"] in ("object", "character", "text"),
            f"{loc}.label must be object, character or text",
        )
        _text(item["caption"], f"{loc}.caption")
        if item["label"] == "text":
            _require("text" in item, f"{loc}: text items require literal text")
        if "text" in item:
            _text(item["text"], f"{loc}.text")
        for key in ("tags", "palette"):
            if key in item:
                _strings(item[key], f"{loc}.{key}")
        compiled_items.append(
            {
                key: (
                    _centered_box(value, width, height, f"{loc}.bbox")
                    if key == "bbox"
                    else value
                )
                for key, value in item.items()
            }
        )

    result = {
        "summary": summary,
        "items": compiled_items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }
    compiled = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return compiled, compiled, width, height


def parse_user_layout(layout_json: str) -> dict:
    """Parse user-supplied JSON string. Raises ValueError on invalid input."""
    _require(isinstance(layout_json, str), "layout_json must be a string")
    stripped = layout_json.strip()
    _require(stripped, "layout_json is empty")
    return json.loads(stripped)