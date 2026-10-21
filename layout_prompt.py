"""BSAI_MOSS_LayoutPromptEnhancer — MOSS-Image layout-aware prompt enhancer.

Inputs:
  raw_prompt       (STRING, multiline): user idea / draft caption
  layout_json      (STRING, multiline, optional): MOSS-Image Layout draft
  target_layout    (COMBO): visual template hint passed to the LLM
  engine           (COMBO): auto / strata_local / nemotron / ninfer / rule_only / none
  language         (COMBO): zh / en
  max_tokens, temperature, ninfer_model, nemotron_model: passthrough to LLM

Outputs:
  positive         (STRING): MOSS-Image-ready prompt (compiled JSON line if a
                            layout was provided, otherwise expanded caption)
  layout_json      (STRING): the validated/compiled layout JSON (empty string
                            if input was a caption-only draft)
  engine_used      (STRING): which engine actually produced the output
  width, height    (INT):    canvas dimensions extracted from the layout
                            (0 / 0 when no layout was provided)

Behaviour:
  - If engine == "none", raw_prompt + layout_json are passed through verbatim
    (still validated/compiled).
  - If the LLM returns plain prose but the user supplied a layout_json, the
    prose is returned as positive — the layout_json input is preserved.
  - If the LLM produces a JSON line on its own, it is validated and compiled
    in-place, overriding the layout_json input.
  - If the user supplies an invalid layout_json and engine != "none", the
    node falls back to caption-only mode and reports the validation error in
    engine_used as "rule_only:invalid_layout".
"""
from __future__ import annotations

import json

from .layout_validate import compile_layout, parse_user_layout
from .llm_backends import (
    call_llm,
    list_nemotron_model_names,
    list_ninfer_model_names,
    list_strata_quant_options,
)

TARGET_LAYOUT_OPTIONS = [
    "free_caption",     # default: prose caption only, no layout
    "single_poster",    # 海报 / 封面 / 单图主视觉
    "two_panel",        # 左右或上下双联画
    "grid",             # 网格知识图 / 信息密集图
    "knowledge_graph",  # 知识图解（标题 + 模块 + 图例）
]

_ENGINE_OPTIONS = ["auto", "strata_local", "nemotron", "ninfer", "rule_only", "none"]


def _build_system_prompt(target_layout: str, language: str) -> str:
    """Build the MOSS-Image specific system prompt fed to the LLM.

    Inherits the "Chinese-first, fall back to English" convention from
    BSAI_FLUX3_PromptEnhancer but adds explicit Layout JSON schema so the LLM
    can produce a valid draft when the user did not supply one.
    """
    layout_hint = {
        "free_caption": (
            "Produce a single cohesive caption (UTF-8 JSON: {\"caption\":\"...\"}). "
            "No layout coordinates."
        ),
        "single_poster": (
            "Produce a Layout draft with summary.caption (global treatment) "
            "and 2-4 items: typically one object (the hero visual) and 1-3 "
            "text items (title, subtitle, supporting copy). Bounding boxes "
            "must be integers in [0, width] x [0, height] with bbox_coord_space='abs' "
            "and bbox_format='xyxy'. Visible text strings must be reproduced verbatim."
        ),
        "two_panel": (
            "Produce a Layout draft with summary.canvas='multi_panel' and 2-4 "
            "items, one panel per logical unit (left/right or top/bottom). "
            "Each panel can contain one object/character and accompanying text."
        ),
        "grid": (
            "Produce a Layout draft with summary.canvas='single' but several "
            "(>=4) items arranged on a coarse grid. Each item gets its own "
            "bbox. Use short captions and short text strings."
        ),
        "knowledge_graph": (
            "Produce a Layout draft with summary.canvas='single' and one text "
            "title, one text subtitle, and 3-6 object/character items acting as "
            "modules or legends. Bounding boxes must be non-overlapping when "
            "feasible; allow small natural overlap."
        ),
    }.get(target_layout, "Produce a single cohesive caption.")

    language_hint = {
        "zh": (
            "captions 写中文，visible text 按用户原始语言保留（中英文/标点/换行都保留）。"
        ),
        "en": (
            "Write captions in English. Preserve visible text verbatim in its "
            "original language, including punctuation, capitalization and line breaks."
        ),
    }.get(language, "Write captions in English.")

    return (
        "You are a MOSS-Image prompt engineer. Your job is to expand a rough "
        "user idea into a MOSS-Image ready prompt.\n"
        "Target model: MOSS-Image (5B single-stream DiT with Layout control).\n"
        f"Language rule: {language_hint}\n"
        f"Layout style: {layout_hint}\n"
        "Schema (Layout draft — exactly these top-level keys):\n"
        "  width: int (>0, multiple of 16)\n"
        "  height: int (>0, multiple of 16)\n"
        "  bbox_coord_space: 'abs'\n"
        "  bbox_format: 'xyxy'\n"
        "  summary: { caption: <nonempty>, meta?: [string], canvas?: 'single'|'multi_panel'|'collage' }\n"
        "  items: [ { label: 'object'|'character'|'text', caption: <nonempty>, bbox: [x1,y1,x2,y2] (ints in [0,width]x[0,height]), text?: <only for label='text'> } ]\n"
        "Constraints:\n"
        " - Do NOT add children/parent/id/elements; the schema above is exhaustive.\n"
        " - Preserve the user's visible wording verbatim in text items.\n"
        " - For ordinary photos/illustrations, prefer a single caption ({\"caption\":\"...\"}).\n"
        " - For posters, multi-panel layouts, knowledge graphics, produce a Layout draft.\n"
        " - Output ONLY the JSON object (draft or caption), no markdown fencing, no commentary."
    )


def _try_parse_layout_response(text: str):
    """If the LLM output looks like a JSON object, try to compile it as a layout."""
    s = (text or "").strip()
    if not s.startswith("{"):
        return None
    try:
        draft = json.loads(s)
    except Exception:
        return None
    if not isinstance(draft, dict):
        return None
    try:
        return compile_layout(draft)
    except Exception:
        return None


class BSAI_MOSS_LayoutPromptEnhancer:
    """Layout-aware prompt enhancer for MOSS-Image.

    Reuses the BSAI LLM backend stack (Strata / Nemotron / NInfer) and the
    rule-only fallback chain. The Layout JSON produced by this node is the
    same ground-truth schema documented in
    `moss-image/skills/moss-image-prompt-enhancer/references/layout.md`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "placeholder": "Rough idea — e.g. \"Make a cozy kitchen poster with a sticky note that reads 'Hello MOSS-Image'\"",
                }),
                "target_layout": (TARGET_LAYOUT_OPTIONS, {
                    "default": "free_caption",
                    "tooltip": "Layout template hint passed to the LLM. Choose 'free_caption' for ordinary photos.",
                }),
                "engine": (_ENGINE_OPTIONS, {"default": "auto"}),
                "language": (["zh", "en"], {"default": "zh"}),
            },
            "optional": {
                "layout_json": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": (
                        "Optional MOSS-Image Layout draft (JSON). When supplied, the LLM is asked to refine it; "
                        "the validated/compiled JSON is returned in the layout_json output. "
                        "Leave empty to let the LLM decide the layout style."
                    ),
                }),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5, "step": 0.05}),
                "ninfer_model": (["(Auto — first model found)"] + list_ninfer_model_names(),
                                 {"default": "(Auto — first model found)",
                                  "tooltip": "NInfer 后端专用：选择要加载的 *.ninfer 模型。"}),
                "nemotron_model": (["(Auto — first model found)"] + list_nemotron_model_names(),
                                   {"default": "(Auto — first model found)",
                                    "tooltip": "Nemotron 后端专用：选择要加载的 GGUF 模型（models/LLM/Nemotron，排除 mmproj）。"}),
                # 追加在末尾，保持旧工作流 widgets_values 索引不变
                "strata_quant": (list_strata_quant_options(), {"default": "Auto (自动适配)",
                    "tooltip": "Strata 后端专用：选择 Qwen3.8-Flash-Next 量化档（Auto 自动适配 + Q2_0/IQ2_XS/IQ3_XXS/IQ3_S 四档）；仅在 Strata 未运行、需自动启动时生效。"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("positive", "layout_json", "engine_used", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "BSAI-MOSS"

    SEARCH_ALIASES = [
        "moss", "moss image", "layout prompt", "moss prompt",
        "layout enhancer", "布局提示词", "排版提示词", "中英长文本",
    ]

    def execute(
        self,
        raw_prompt,
        target_layout="free_caption",
        engine="auto",
        language="zh",
        layout_json="",
        max_tokens=1024,
        temperature=0.7,
        ninfer_model="(Auto — first model found)",
        nemotron_model="(Auto — first model found)",
        strata_quant="Auto (自动适配)",
    ):
        raw_prompt = (raw_prompt or "").strip()
        layout_json = (layout_json or "").strip()
        sys_prompt = _build_system_prompt(target_layout, language)

        # Step 1 — parse user-supplied layout_json (if any)
        user_layout_draft = None
        user_layout_error = ""
        if layout_json:
            try:
                user_layout_draft = parse_user_layout(layout_json)
            except Exception as e:
                user_layout_error = f"{type(e).__name__}: {e}"

        # Step 2 — none engine: pure passthrough, still validate user layout
        if engine == "none":
            if user_layout_draft is not None:
                try:
                    prose, compiled, w, h = compile_layout(user_layout_draft)
                    return (prose, compiled, "none", w, h)
                except Exception as e:
                    return (raw_prompt, "", f"none:invalid_layout:{e}", 0, 0)
            return (raw_prompt, "", "none", 0, 0)

        # Step 3 — prepare the LLM user message. When a user layout is supplied
        # we ask the LLM to refine it; otherwise we ask the LLM to draft from
        # raw_prompt under the target_layout hint.
        if user_layout_draft is not None:
            user_msg = (
                "The user supplied this MOSS-Image Layout draft (JSON). "
                "Refine captions, palette and box geometry while preserving "
                "the user's intent and any visible text strings. Return the "
                "complete revised Layout JSON.\n\n"
                f"```json\n{json.dumps(user_layout_draft, ensure_ascii=False, indent=2)}\n```"
            )
        else:
            user_msg = (
                f"User idea: {raw_prompt}\n\n"
                "Produce the MOSS-Image prompt (Layout draft or single caption) "
                "that best realises the brief. Honour the language rule."
            )

        # Step 4 — call the LLM
        engine_used, llm_text = call_llm(
            engine, user_msg, sys_prompt,
            max_tokens=max_tokens, temperature=temperature,
            ninfer_model=ninfer_model, nemotron_model=nemotron_model,
            strata_quant=strata_quant,
        )

        # Step 5 — if the LLM returned a layout draft, compile it.
        compiled_layout = None
        compiled_dims = (0, 0)
        llm_layout = _try_parse_layout_response(llm_text)
        if llm_layout is not None:
            compiled_layout = llm_layout[1]  # already-compacted JSON string
            compiled_dims = (llm_layout[2], llm_layout[3])

        # Step 6 — final assembly
        if compiled_layout is not None:
            return (compiled_layout, compiled_layout, engine_used,
                    compiled_dims[0], compiled_dims[1])

        # No layout came out of the LLM — treat as caption (even when the
        # user supplied a layout, this means the LLM could not refine it).
        if user_layout_draft is not None and not user_layout_error:
            # Surface the user's layout as the validated output instead of dropping it
            try:
                _, compiled = compile_layout(user_layout_draft)
                w, h = compiled_dims_for(user_layout_draft)
                return (compiled, compiled, engine_used + ":user_layout_passthrough",
                        w, h)
            except Exception:
                pass
        if user_layout_error:
            return (llm_text or raw_prompt, "",
                    engine_used + ":invalid_user_layout:" + user_layout_error, 0, 0)

        return (llm_text or raw_prompt, "", engine_used, 0, 0)


def compiled_dims_for(draft: dict):
    """Tiny helper: return (width, height) for a validated layout draft."""
    return (int(draft.get("width", 0)), int(draft.get("height", 0)))


NODE_CLASS_MAPPINGS = {
    "BSAI_MOSS_LayoutPromptEnhancer": BSAI_MOSS_LayoutPromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_MOSS_LayoutPromptEnhancer": "BSAI MOSS Layout Prompt Enhancer",
}