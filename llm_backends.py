"""Shared LLM backend dispatcher for BSAI-MOSS-ComfyUI.

Mirrors the dispatch pattern from BSAI-FLUX3-ComfyUI/enhancer.py but adapted
to the MOSS-Image Layout Prompt use case (the dispatcher only ever produces
ONE positive string per call; it doesn't need fallback chains or rule_only
generation — that's the caller's job).

Engine semantics:
  - "auto"        → try ninfer → strata_local → nemotron → rule_only → none
  - "strata_local"→ only the local Strata server (127.0.0.1:8080)
  - "nemotron"    → only the local Nemotron llama.cpp server (127.0.0.1:8085)
  - "ninfer"      → only the bundled ninfer-serve.exe on 127.0.0.1:8082
  - "rule_only"   → skip LLM and return the rule-only enhanced prompt
  - "none"        → pass raw_prompt through verbatim

Returns (engine_used, positive_string) on success, or raises RuntimeError
after every LLM backend fails. The caller decides whether to fall back to
rule_only or surface the error.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys
import traceback
from typing import Optional, Tuple


_HERE = os.path.dirname(os.path.abspath(__file__))
_FLUX3_DIR = os.path.normpath(os.path.join(_HERE, "..", "BSAI-FLUX3-ComfyUI"))
_STRATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "BSAI-LLM-Strata"))


def _ensure_path(p: str) -> None:
    if p and p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)


_ensure_path(_FLUX3_DIR)
_ensure_path(_STRATA_DIR)


def _load_strata():
    try:
        from strata_client import strata_health, strata_chat  # type: ignore
        from strata_launcher import ensure_strata  # type: ignore
        return strata_health, strata_chat, ensure_strata
    except Exception as e:
        print(f"[BSAI-MOSS] strata_client unavailable: {type(e).__name__}: {e}")
        return None


def _load_ninfer():
    try:
        from ninfer_client import (  # type: ignore
            ninfer_health, ninfer_chat, ensure_ninfer, list_ninfer_model_names,
        )
        return ninfer_health, ninfer_chat, ensure_ninfer, list_ninfer_model_names
    except Exception as e:
        print(f"[BSAI-MOSS] ninfer_client unavailable: {type(e).__name__}: {e}")
        return None


def _load_nemotron_manager():
    """Load NemotronManager from BSAI-LLM-Strata/strata_manager.py (importlib)."""
    try:
        manager_py = os.path.join(_STRATA_DIR, "strata_manager.py")
        if not os.path.exists(manager_py):
            return None
        spec = _ilu.spec_from_file_location("bsai_strata_manager_for_moss", manager_py)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.NemotronManager
    except Exception as e:
        print(f"[BSAI-MOSS] NemotronManager unavailable: {type(e).__name__}: {e}")
        return None


def _load_strata_manager_mod():
    """Load strata_manager module (StrataManager + QUANT_OPTIONS) via importlib."""
    try:
        manager_py = os.path.join(_STRATA_DIR, "strata_manager.py")
        if not os.path.exists(manager_py):
            return None
        spec = _ilu.spec_from_file_location("bsai_strata_manager_for_moss_mod", manager_py)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"[BSAI-MOSS] strata_manager unavailable: {type(e).__name__}: {e}")
        return None


# Cached lazy loaders
_STRATA = _load_strata()
_NINFER = _load_ninfer()
_NEMOTRON = _load_nemotron_manager()
_STRATA_MGR_MOD = _load_strata_manager_mod()


def list_strata_quant_options() -> list:
    """Strata 引擎量化档下拉：Auto + Qwen3.8-Flash-Next 的 4 个量化档。"""
    mod = _STRATA_MGR_MOD or _load_strata_manager_mod()
    if mod is None:
        return ["Auto (自动适配)"]
    try:
        opts = list(mod.QUANT_OPTIONS.keys())
        return opts or ["Auto (自动适配)"]
    except Exception:
        return ["Auto (自动适配)"]


def list_nemotron_model_names() -> list:
    """Nemotron 模型下拉：models/LLM/Nemotron 下所有非 mmproj 的 *.gguf（相对路径）。

    与其他 BSAI 插件一致，直接复用 NemotronManager.list_models()，
    保证返回值能被 NemotronManager.resolve_model() 正确解析。
    """
    try:
        manager_py = os.path.join(_STRATA_DIR, "strata_manager.py")
        if not os.path.exists(manager_py):
            return []
        spec = _ilu.spec_from_file_location(
            "bsai_strata_manager_for_moss_list", manager_py
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return [rel for rel, _abs, kind in mod.NemotronManager.list_models()
                if kind == "model"]
    except Exception as e:
        print(f"[BSAI-MOSS] list_nemotron_model_names failed: {type(e).__name__}: {e}")
        return []


def list_ninfer_model_names() -> list:
    if _NINFER is None:
        return []
    _, _, _, list_fn = _NINFER
    try:
        return list_fn()
    except Exception:
        return []


def call_llm(
    engine: str,
    raw_prompt: str,
    sys_prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    ninfer_model: str = "(Auto — first model found)",
    nemotron_model: str = "(Auto — first model found)",
    strata_quant: str = "Auto (自动适配)",
) -> Tuple[str, str]:
    """Dispatch the LLM call and return (engine_used, positive_text).

    engine_used is one of "ninfer" | "strata_local" | "nemotron" | "rule_only" | "none".
    Raises RuntimeError if every LLM backend fails (caller falls back to rule_only).
    """
    engine_norm = (engine or "auto").strip().lower()

    if engine_norm == "none":
        return "none", raw_prompt

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": raw_prompt},
    ]

    order = []
    if engine_norm == "auto":
        order = ["ninfer", "strata_local", "nemotron"]
    elif engine_norm in ("ninfer", "strata_local", "nemotron"):
        order = [engine_norm]
    elif engine_norm == "rule_only":
        return "rule_only", raw_prompt
    else:
        # Unknown engine string → behave like auto
        order = ["ninfer", "strata_local", "nemotron"]

    last_err: Optional[Exception] = None
    for which in order:
        try:
            if which == "ninfer" and _NINFER is not None:
                _, ninfer_chat, ensure_ninfer, _ = _NINFER
                model_name = (
                    None
                    if str(ninfer_model).startswith("(Auto")
                    else ninfer_model
                )
                ensure_ninfer(model_name=model_name)
                text = ninfer_chat(
                    messages, max_tokens=max_tokens, temperature=temperature,
                    model_name=model_name,
                )
                return "ninfer", (text or "").strip() or raw_prompt

            if which == "strata_local" and _STRATA is not None:
                strata_health, strata_chat, ensure_strata = _STRATA
                if not strata_health():
                    # 优先走 StrataManager（支持 strata_quant 量化档选择），
                    # 失败时回退旧 launcher（固定 Q2_0）
                    _started = False
                    if _STRATA_MGR_MOD is not None:
                        try:
                            _ok, _msg = _STRATA_MGR_MOD.StrataManager.ensure_running(
                                quant_label=strata_quant or "Auto (自动适配)")
                            _started = bool(_ok)
                            if not _ok:
                                print(f"[BSAI-MOSS] StrataManager auto-start failed: {_msg}")
                        except Exception as _e:
                            print(f"[BSAI-MOSS] StrataManager error: {type(_e).__name__}: {_e}")
                    if not _started and not strata_health():
                        ensure_strata(log=print)
                if strata_health():
                    text = strata_chat(
                        messages, max_tokens=max_tokens, temperature=temperature,
                    )
                    return "strata_local", (text or "").strip() or raw_prompt

            if which == "nemotron" and _NEMOTRON is not None:
                model_path = (
                    None
                    if str(nemotron_model).startswith("(Auto")
                    else nemotron_model
                )
                ok, _msg = _NEMOTRON.ensure_running(model_path=model_path)
                if ok:
                    mids = _NEMOTRON.models()
                    r = _NEMOTRON.chat(
                        messages,
                        model=(mids[0] if mids else None),
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    text = r["choices"][0]["message"]["content"].strip()
                    return "nemotron", text or raw_prompt
        except Exception as e:
            last_err = e
            print(
                f"[BSAI-MOSS] {which} failed: {type(e).__name__}: {e}",
                flush=True,
            )
            continue

    if last_err is not None:
        raise RuntimeError(
            f"All LLM backends failed; last error: {type(last_err).__name__}: {last_err}"
        ) from last_err
    # No backend available at all
    raise RuntimeError(
        "No LLM backend available; install Strata/Nemotron/NInfer or use engine='rule_only'."
    )