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


class MOSSImageDMDApply:
    """对 MODEL 应用 DMD 主加速 LoRA（等价于 LoraLoaderModelOnly）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "MOSS-Image/DMD"

    def apply(self, model, lora_name, strength):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            raise RuntimeError("LoRA 文件不存在: %s" % lora_name)
        lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        patcher, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, strength, 0.0)
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
            }
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
               width, height, seed, steps):
        patcher = model
        for name in (time_embedding_lora, dataspace_lora):
            patcher = self._apply_lora(patcher, name)

        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))

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
