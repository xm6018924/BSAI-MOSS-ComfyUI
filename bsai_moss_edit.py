# -*- coding: utf-8 -*-
"""BSAI MOSS-Image Edit —— 参考图编辑节点（跟进内核 #17296 原生 MOSS-Image edit LoRA 支持）。

内核 60aade0（Comfy-Org/ComfyUI）为 MOSSImage 增加了 reference_latents /
reference_latents_method / ref_kv 传递链，并新增官方节点 TextEncodeMOSSImageEdit。
本模块在 BSAI-MOSS-ComfyUI 内提供同能力的一站式 BSAI 节点：

  * BSAIMOSSImageEditEncode
      挂载 MOSS-Image 编辑 LoRA（可选）+ RoPE 选项 + 参考图 VAE 编码 +
      正/负条件一次性输出，直接进 KSampler。参考图布局三选一：
        - ai-toolkit                  (index_timestep_zero_isolated, 文本编码器看参考图, ref_kv 一次计算)
        - ai-toolkit (kv_cache off)   (index_timestep_zero,        refs 每步重算)
        - Moss-Image-Edit             (index_timestep_zero_text_start, 文本编码器不看参考图, ref_kv 一次计算)

与内核节点的差异：
  * 旧式 INPUT_TYPES 风格，与 BSAI 插件家族（layout_prompt / dmd_nodes / int8_probe）一致，
    绘世/ComfyUI-Manager 直接识别。
  * 多参考图用固定 4 路可选输入（reference_1..reference_4），不依赖新 io.Autogrow API。
  * 正/负条件同批参考图编码，一个节点完成整条编辑链路。
"""

import math

import torch

import folder_paths
import comfy.sd
import comfy.utils
import node_helpers

# 内核 nodes_moss_image.EDIT_FORMATS 三布局复刻
EDIT_FORMATS = {
    "ai-toolkit": ("index_timestep_zero_isolated", True),
    "ai-toolkit (kv_cache off)": ("index_timestep_zero", True),
    "Moss-Image-Edit": ("index_timestep_zero_text_start", False),
}

_VISION_MARK = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
_VISION_MAX_PX = 384  # 内核将参考图缩到至多 384x384 给文本编码器


def _gather_references(refs):
    """收集非空参考图，按输入名排序（与内核 TextEncodeMOSSImageEdit.execute 一致）。"""
    return [refs[name] for name in sorted(refs) if refs.get(name) is not None]


def _encode_with_references(clip, prompt, vae, images, method, vision):
    """复刻内核 TextEncodeMOSSImageEdit.execute 的编码逻辑（60aade0）。"""
    vl = []
    if vision:
        # 文本编码器看到的是缩到 <=384x384 的参考图，前置 "Picture N:" 标记
        for image in images:
            scale = min(1.0, math.sqrt(_VISION_MAX_PX * _VISION_MAX_PX /
                                       (image.shape[1] * image.shape[2])))
            vl.append(comfy.utils.common_upscale(
                image.movedim(-1, 1), round(image.shape[2] * scale),
                round(image.shape[1] * scale), "area", "disabled").movedim(1, -1))
        prompt = "".join(_VISION_MARK.format(i + 1) for i in range(len(vl))) + prompt
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=vl))
    refs = [vae.encode(image[:, :, :, :3]) for image in images]
    return node_helpers.conditioning_set_values(
        conditioning, {"reference_latents": refs,
                       "reference_latents_method": method})


def _apply_rope_options(patcher, rope_enabled, scale_x, shift_x, scale_y, shift_y,
                        scale_t, shift_t):
    """RoPE 缩放/平移（复用 dmd_nodes 语义，核心 #17172）。"""
    if not rope_enabled:
        return patcher, ""
    patcher = patcher.clone()
    patcher.set_model_rope_options(scale_x, shift_x, scale_y, shift_y, scale_t, shift_t)
    return patcher, (f"RoPE: scale_x={scale_x} shift_x={shift_x} scale_y={scale_y} "
                     f"shift_y={shift_y} shift_t={shift_t}")


class BSAIMOSSImageEditEncode:
    """MOSS-Image 编辑工作流一站式节点：编辑 LoRA + RoPE + 参考图编码 + 正/负条件。"""

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "edit_format": (list(EDIT_FORMATS),
                                {"default": "ai-toolkit",
                                 "tooltip": "编辑 LoRA 的训练布局：ai-toolkit 的 MOSS-Image Edit "
                                            "Training（kv_cache 开/关）或 Moss-Image-Edit。"
                                            "kv_cache off 时参考图每步重算。"}),
                "edit_lora_name": (loras, {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                            "step": 0.01}),
            },
            "optional": {
                "reference_1": ("IMAGE",),
                "reference_2": ("IMAGE",),
                "reference_3": ("IMAGE",),
                "reference_4": ("IMAGE",),
                "rope_enabled": ("BOOLEAN", {"default": False,
                                             "label_on": "启用", "label_off": "关闭",
                                             "tooltip": "启用 MOSS-Image RoPE 缩放/平移。"}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_x": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_y": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
                "scale_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                      "tooltip": "MOSS-Image 不使用 scale_t，仅为与官方节点同构保留。"}),
                "shift_t": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "note")
    FUNCTION = "encode"
    CATEGORY = "MOSS-Image/Edit"

    def encode(self, model, clip, vae, prompt, negative, edit_format,
               edit_lora_name="None", lora_strength=1.0,
               reference_1=None, reference_2=None, reference_3=None, reference_4=None,
               rope_enabled=False, scale_x=1.0, shift_x=0.0, scale_y=1.0,
               shift_y=0.0, scale_t=1.0, shift_t=0.0):
        notes = []

        # 1) 可选编辑 LoRA 挂载（作用于 UNET）
        patcher = model
        if edit_lora_name not in ("None", "", None):
            lora_path = folder_paths.get_full_path("loras", edit_lora_name)
            if not lora_path:
                raise RuntimeError("LoRA 文件不存在: %s" % edit_lora_name)
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            patcher, _ = comfy.sd.load_lora_for_models(patcher, None, lora_sd,
                                                       lora_strength, 0.0)
            notes.append("LoRA: %s x%.3f" % (edit_lora_name, lora_strength))

        # 2) 可选 RoPE
        patcher, rope_note = _apply_rope_options(patcher, rope_enabled, scale_x,
                                                 shift_x, scale_y, shift_y,
                                                 scale_t, shift_t)
        if rope_note:
            notes.append(rope_note)

        # 3) 参考图 + 条件编码
        refs = _gather_references({"reference_1": reference_1, "reference_2": reference_2,
                                   "reference_3": reference_3, "reference_4": reference_4})
        if not refs:
            raise RuntimeError("MOSS-Image Edit 至少需要 1 张参考图（reference_1 起）")
        method, vision = EDIT_FORMATS[edit_format]
        positive = _encode_with_references(clip, prompt, vae, refs, method, vision)
        negative = _encode_with_references(clip, negative, vae, refs, method, vision)
        notes.append("format=%s refs=%d vision=%s" % (edit_format, len(refs), vision))

        return (patcher, positive, negative, " | ".join(notes))


NODE_CLASS_MAPPINGS = {"BSAIMOSSImageEditEncode": BSAIMOSSImageEditEncode}
NODE_DISPLAY_NAME_MAPPINGS = {"BSAIMOSSImageEditEncode": "BSAI MOSS-Image Edit Encode"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
