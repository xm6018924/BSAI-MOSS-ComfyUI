"""BSAI-MOSS-ComfyUI — MOSS-Image day-0+ companion plugin for ComfyUI.

Adds two BSAI-native nodes that complement the official `moss-image` plugin
shipped by OpenMOSS:

  * BSAI_MOSS_LayoutPromptEnhancer
      Layout-aware prompt enhancer. Takes a rough idea + optional MOSS-Image
      Layout JSON draft, dispatches to the BSAI LLM backend stack
      (Strata / Nemotron / NInfer) and returns a MOSS-Image-ready positive
      prompt (compiled JSON line if a layout was supplied, otherwise a
      caption). The compiled JSON is `abs_centralized` + `yxyx` with integer
      boxes, matching the 2026-10-19 MOSS-Image technical report.

  * BSAI_MOSS_SamplerPresets
      Wrap the technical report's recommended sampling recipe as a single
      combo: sampler=euler, scheduler=flow, 50 steps, CFG 3.0, shift 5, with
      alternate presets for the SFT-4steps-turbo LoRA, long-text rendering
      (60 steps / CFG 3.5), and low-VRAM (35 steps / CFG 2.5).

  * MOSSImageDMDApply / MOSSImageDMDDataspaceSampler (dmd_nodes.py)
      DMD 蒸馏加速节点（原 BSAI-MOSS-DMD 独立插件，已并入本插件）。
      对 MODEL 应用主加速 LoRA，以及 time_embedding + dataspace 双 LoRA
      少步采样（euler / simple / CFG 1.0，与 V5 工作流 8 步 DMD 分支一致）。

Self-contained: shares LLM client code paths with BSAI-FLUX3-ComfyUI via
importlib rather than cross-plugin imports. Falls back to rule-only or
passthrough (engine="none") when no LLM backend is available.
"""
from .layout_prompt import NODE_CLASS_MAPPINGS as _A, NODE_DISPLAY_NAME_MAPPINGS as _AD
from .sampler_presets import NODE_CLASS_MAPPINGS as _B, NODE_DISPLAY_NAME_MAPPINGS as _BD
from .dmd_nodes import NODE_CLASS_MAPPINGS as _C, NODE_DISPLAY_NAME_MAPPINGS as _CD

NODE_CLASS_MAPPINGS = {**_A, **_B, **_C}
NODE_DISPLAY_NAME_MAPPINGS = {**_AD, **_BD, **_CD}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]