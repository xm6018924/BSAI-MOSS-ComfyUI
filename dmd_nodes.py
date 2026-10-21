# -*- coding: utf-8 -*-
"""BSAI MOSS-Image DMD 加速节点（原 BSAI-MOSS-DMD 独立插件，已并入 BSAI-MOSS-ComfyUI）。

节点类名保持与旧工作流一致（MOSSImageDMDApply / MOSSImageDMDDataspaceSampler），
旧 JSON 可直接加载。MOSS latent 规格：128 通道 @ H//16（与官方 EmptyFlux2LatentImage 一致）。
"""

import torch

import folder_paths
import comfy.sd
import comfy.utils
import comfy.sample
import comfy.model_management as mm
import node_helpers


def _apply_rope_options(patcher, rope_enabled, scale_x, shift_x, scale_y, shift_y,
                        scale_t, shift_t):
    """可选 RoPE 缩放/平移（核心 #17172）。

    moss_image 的 rope() 从 transformer_options["rope_options"] 读取
    scale_x/scale_y/shift_x/shift_y/shift_t（scale_t 对 MOSS 无效，仅为
    与官方 ScaleROPE 节点同构保留）。关闭时保持旧行为。
    """
    if not rope_enabled:
        return patcher, ""
    patcher = patcher.clone()
    patcher.set_model_rope_options(scale_x, shift_x, scale_y, shift_y, scale_t, shift_t)
    note = (f"RoPE: scale_x={scale_x} shift_x={shift_x} scale_y={scale_y} "
            f"shift_y={shift_y} shift_t={shift_t}")
    return patcher, note


def _apply_moss_control(positive, negative, vae, color_map, depth_map,
                        strength, start_percent, end_percent):
    """对 conditioning 附加 MOSS-Image 色彩/深度控制（核心 #17206 语义）。

    复刻官方 MOSSImageControl.execute：双图 VAE 编码后沿通道拼接
    [color | depth]，缺失图为全零 latent；嵌入器线性且零即无控制，
    因此缩放 latent 即缩放控制贡献。
    """
    latents = [vae.encode(m[:, :, :, :3]) if m is not None else None
               for m in (color_map, depth_map)]
    ref = next((latent for latent in latents if latent is not None), None)
    if ref is None:
        return positive, negative, ""
    batch = max(latent.shape[0] for latent in latents if latent is not None)
    latents = [torch.zeros_like(ref) if latent is None
               else latent if latent.shape[-2:] == ref.shape[-2:]
               else comfy.utils.common_upscale(latent, ref.shape[-1], ref.shape[-2],
                                               "bilinear", "center")
               for latent in latents]
    control = torch.cat([comfy.utils.resize_to_batch_size(latent, batch)
                         for latent in latents], dim=1) * strength
    values = {"concat_latent_image": control}
    if positive:
        positive = node_helpers.conditioning_set_values_with_timestep_range(
            positive, values, start_percent, end_percent)
    if negative:
        negative = node_helpers.conditioning_set_values_with_timestep_range(
            negative, values, start_percent, end_percent)
    kinds = ("color" if color_map is not None else "") + \
            ("+depth" if depth_map is not None else "")
    note = (f"控制: {kinds} strength={strength} 范围 {start_percent:.3f}-{end_percent:.3f} "
            f"latent={tuple(control.shape)}")
    return positive, negative, note


class MOSSImageDMDApply:
    """对 MODEL 应用 DMD 主加速 LoRA（等价于 LoraLoaderModelOnly）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "rope_enabled": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "关闭",
                                             "tooltip": "启用 MOSS-Image RoPE 缩放/平移（超宽/超长画幅或构图平移）。"}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_x": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_y": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
                "scale_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                      "tooltip": "MOSS-Image 不使用 scale_t，仅为与官方 ScaleROPE 节点同构保留。"}),
                "shift_t": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MOSS-Image/DMD"

    def apply(self, model, lora_name, strength, rope_enabled=False, scale_x=1.0,
              shift_x=0.0, scale_y=1.0, shift_y=0.0, scale_t=1.0, shift_t=0.0):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise RuntimeError("LoRA 文件不存在: %s" % lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        patcher, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, strength, 0.0)
        patcher, rope_note = _apply_rope_options(patcher, rope_enabled, scale_x, shift_x,
                                                 scale_y, shift_y, scale_t, shift_t)
        if rope_note:
            print("[MOSSImageDMDApply] %s" % rope_note, flush=True)
        return (patcher,)


class MOSSImageDMDDataspaceSampler:
    """DMD 少步蒸馏采样：内置 time_embedding + dataspace 双 LoRA，输出 LATENT。

    控件顺序（与旧工作流 widgets_values 一一对应）：
      prompt, time_embedding_lora, dataspace_lora, width, height, seed(+control), steps
    采样参数沿用 V5 工作流 DMD 分支实测配置：euler + simple + CFG 1.0。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "time_embedding_lora": (folder_paths.get_filename_list("loras"),),
                "dataspace_lora": (folder_paths.get_filename_list("loras"),),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
            },
            "optional": {
                # 色彩/深度控制（核心 #17206）：提供 VAE + 任一图即启用
                "vae": ("VAE", {"tooltip": "色彩/深度控制所需的 VAE；不接控制图可留空。"}),
                "color_map": ("IMAGE", {"tooltip": "区域色彩图（每区域一种平色，输出尺寸）。"}),
                "depth_map": ("IMAGE", {"tooltip": "深度图（近处为白，输出尺寸）。"}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "control_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "control_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                # RoPE 缩放/平移（核心 #17172）
                "rope_enabled": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "关闭",
                                             "tooltip": "启用 MOSS-Image RoPE 缩放/平移（超宽/超长画幅或构图平移）。"}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_x": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_y": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
                "scale_t": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                      "tooltip": "MOSS-Image 不使用 scale_t，仅为与官方 ScaleROPE 节点同构保留。"}),
                "shift_t": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "MOSS-Image/DMD"

    def _apply_lora(self, patcher, lora_name, strength=1.0):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise RuntimeError("LoRA 文件不存在: %s" % lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        patcher, _ = comfy.sd.load_lora_for_models(patcher, None, lora_sd, strength, 0.0)
        return patcher

    def sample(self, model, clip, prompt, time_embedding_lora, dataspace_lora,
               width, height, seed, steps, vae=None, color_map=None, depth_map=None,
               control_strength=1.0, control_start_percent=0.0, control_end_percent=1.0,
               rope_enabled=False, scale_x=1.0, shift_x=0.0, scale_y=1.0, shift_y=0.0,
               scale_t=1.0, shift_t=0.0):
        patcher = model
        for name in (time_embedding_lora, dataspace_lora):
            patcher = self._apply_lora(patcher, name)
        patcher, rope_note = _apply_rope_options(patcher, rope_enabled, scale_x, shift_x,
                                                 scale_y, shift_y, scale_t, shift_t)

        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))

        if vae is not None and (color_map is not None or depth_map is not None):
            cond, _, control_note = _apply_moss_control(
                cond, [], vae, color_map, depth_map,
                control_strength, control_start_percent, control_end_percent)
        elif vae is None and (color_map is not None or depth_map is not None):
            raise RuntimeError("提供了色彩/深度控制图但未接 VAE — 控制图需要 VAE 编码为 latent。")
        else:
            control_note = ""

        for note in (rope_note, control_note):
            if note:
                print("[MOSSImageDMDDataspaceSampler] %s" % note, flush=True)

        dev = mm.get_torch_device()
        # MOSS-Image latent: 128 通道 @ H//16（Flux2 式）
        latent = torch.zeros((1, 128, height // 16, width // 16), device=dev, dtype=torch.bfloat16)
        noise = comfy.sample.prepare_noise(latent, seed, None)
        out = comfy.sample.sample(patcher, noise, steps, 1.0, "euler", "simple",
                                  cond, [], latent_image={"samples": latent},
                                  denoise=1.0, seed=seed)
        return ({"samples": out},)


NODE_CLASS_MAPPINGS = {
    "MOSSImageDMDApply": MOSSImageDMDApply,
    "MOSSImageDMDDataspaceSampler": MOSSImageDMDDataspaceSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSImageDMDApply": "MOSS-Image DMD Apply (加速LoRA)",
    "MOSSImageDMDDataspaceSampler": "MOSS-Image DMD Dataspace Sampler (4/8步加速)",
}
