# -*- coding: utf-8 -*-
"""
BSA MOSS 多量化格式探测节点 (BSAMOSSInt8Probe)
====================================================================
基于官方 ComfyUI v0.42 新量化栈（NVFP4 / MXFP8 / FP8 / INT8，GPU 侧已支持）
的 MOSS-Image 端到端探测：选择量化 UNET + 量化 TE + VAE，走官方加载链路
（detect_layer_quantization / convert_old_quants / mixed_precision_ops 自动
生效），完成采样与解码，输出诊断信息 + IMAGE。

量化类型自动判定（打印到控制台）：
  - comfy_quant v1 元数据键（*.comfy_quant）          → 新一代量化
  - scaled_fp8 键（model.diffusion_model.scaled_fp8）  → FP8 scaled
  - 权重 dtype 分布（float8_e4m3fn / int8 / nvfp4）     → 兜底判定
  - 无上述标记                                              → BF16/FP16 原版

文件下拉全部走 folder_paths 动态扫描（含子目录），不再硬编码路径。

注：本节点由原独立插件 BSAI-MOSS-INT8-Probe 并入 BSAI-MOSS-ComfyUI，
类名 BSAMOSSInt8Probe 保持不变，已有工作流可直接加载。
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy import utils, sd
from comfy.sd import VAE, CLIPType
import comfy.model_management as mm
import comfy.sample as csample

# 兼容旧目录布局：ROOT 仅用于无子目录时的兜底（现代 ComfyUI 直接用 folder_paths）
ROOT = os.path.dirname(folder_paths.base_path)


def _load_layout_compiler():
    """从同插件 layout_validate 模块加载布局校验/编译（并入后直接包内导入）。"""
    try:
        from .layout_validate import compile_layout
        return compile_layout
    except Exception as _e:
        print(f"[BSAI-MOSS-ComfyUI/int8_probe] layout validator unavailable: "
              f"{type(_e).__name__}: {_e}")
        return None


_compile_layout = _load_layout_compiler()


def _load_tespeed():
    """懒加载 TE-Speed-MOSS 节点（模型级步预测缓存加速，50步场景）。

    返回 TESpeedMoss 节点类；插件未安装或加载失败时返回 None（探针不依赖它）。
    """
    try:
        import importlib.util as _ilu
        for _cn_dir in folder_paths.get_folder_paths("custom_nodes"):
            _p = os.path.join(_cn_dir, "TE-Speed-MOSS", "__init__.py")
            if not os.path.exists(_p):
                continue
            _spec = _ilu.spec_from_file_location("bsai_moss_tespeed_loader", _p)
            _m = _ilu.module_from_spec(_spec)
            sys.modules["bsai_moss_tespeed_loader"] = _m
            _spec.loader.exec_module(_m)
            return _m.NODE_CLASS_MAPPINGS.get("TESpeedMoss")
        return None
    except Exception as _e:
        print(f"[BSAI-MOSS-ComfyUI/int8_probe] TE-Speed-MOSS unavailable: "
              f"{type(_e).__name__}: {_e}")
        return None


_tespeed_cls = _load_tespeed()


def _detect_quant(sd_state, label):
    """判定权重文件的量化类型，返回 (kind, detail)。"""
    detail = []
    found = []
    # 1) 新一代量化元数据 v1
    n_quant = sum(1 for k in sd_state if k.endswith(".comfy_quant"))
    if n_quant:
        found.append("comfy_quant v1(%d层)" % n_quant)
    # 2) 旧式 scaled_fp8
    for key in ("model.diffusion_model.scaled_fp8", "diffusion_model.scaled_fp8", "scaled_fp8"):
        if key in sd_state:
            found.append(key + " → FP8 scaled")
            break
    # 3) dtype 分布兜底
    from collections import Counter
    dtypes = Counter(v.dtype.__str__() if hasattr(v, "dtype") else "?" for v in sd_state.values()
                     if hasattr(v, "dtype"))
    top = dtypes.most_common(4)
    dtxt = ", ".join("%s x%d" % (d, c) for d, c in top)
    detail.append("dtype: " + dtxt)
    if any("float8" in d for d, _ in top):
        found.append("FP8")
    if any("int8" in d for d, _ in top):
        found.append("INT8")
    if any("nvfp4" in d.lower() for d, _ in top):
        found.append("NVFP4")
    if any("mxfp" in d.lower() for d, _ in top):
        found.append("MXFP8")
    kind = " + ".join(found) if found else "BF16/FP16(未量化)"
    return kind, "; ".join(detail)


class BSAMOSSInt8Probe:
    """MOSS-Image 多量化探测：官方加载链路 + 量化诊断。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet": (folder_paths.get_filename_list("diffusion_models"),),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
                "vae": (folder_paths.get_filename_list("vae"),),
                "lora_name": (["（无）"] + folder_paths.get_filename_list("loras"), {
                    "tooltip": "加速/风格 LoRA（如 MOSS-Image-SFT-4steps-turbo）：选（无）则不加载，用于对比有无加速 LoRA 的采样质量与耗时。",
                }),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cozy kitchen with a bowl of fruit on the counter, a potted plant, a white sticky note on the fridge that reads: Hello MOSS-Image, photorealistic, 8k",
                    "placeholder": "Enter your test prompt here...",
                }),
                "layout_json": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Optional MOSS-Image Layout draft (JSON). Validated + compiled via BSAI-MOSS-ComfyUI/layout_validate; the compiled JSON is appended to the prompt so the quantized UNET is exercised with a layout-shaped input.",
                }),
                "steps": ("INT", {"default": 25, "min": 1, "max": 4096}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 16}),
            },
            "optional": {
                # 色彩/深度控制探测（核心 #17206）：用已加载的 MOSS VAE 编码
                "color_map": ("IMAGE", {"tooltip": "区域色彩图（每区域一种平色，输出尺寸）。"}),
                "depth_map": ("IMAGE", {"tooltip": "深度图（近处为白，输出尺寸）。"}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "control_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "control_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                # RoPE 缩放/平移探测（核心 #17172）
                "rope_enabled": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "关闭",
                                             "tooltip": "启用 MOSS-Image RoPE 缩放/平移并纳入诊断。"}),
                "scale_x": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_x": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
                "scale_y": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_y": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
                "shift_t": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 256.0, "step": 0.1}),
                # ---- 内置 TE-Speed MOSS 50步加速（模型级步预测缓存）----
                "te_speed_enabled": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "关闭",
                                                 "tooltip": "内置 TE-Speed MOSS 加速：对已加载 UNET 应用步预测缓存（模型包装，不改变原生注意力/采样器）。启用后无需在工作流中额外挂 TE-Speed MOSS 节点；50步场景收益最大。"}),
                "te_speed_attention": (["default", "sageattn", "sdpa", "flashattn", "kitchen_int8", "kitchen_sol"],
                                       {"default": "default",
                                        "tooltip": "Attention backend。default 保持 ComfyUI 原生注意力（对 MOSS-Image 已是最快路径）；融合/稀疏后端更慢且改变输出。"}),
                "te_speed_step_cache": (["te_predictor", "te_predictor 2"], {"default": "te_predictor 2",
                                        "tooltip": "te_predictor: 线性两点预测；te_predictor 2: 对最近四次真实输出最小二乘拟合，适配 40+ 步计划（推荐）。"}),
                "te_speed_reuse_threshold": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 3.0, "step": 0.01,
                                        "tooltip": "允许预测的相对 latent 变化阈值（0.12 匹配 MOSS euler 实测步间变化）。"}),
                "te_speed_start_percent": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "te_speed_end_percent": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
                "te_speed_predictor_error_limit": ("FLOAT", {"default": 0.06, "min": 0.01, "max": 1.0, "step": 0.01,
                                        "tooltip": "预测偏差触发冷却阈值（0.06 保守默认；0.08 为发布基准值；0.05 在 50 步保留更多细节）。"}),
                "te_speed_uncond_cache": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "关闭",
                                        "tooltip": "对负向/无条件分支独立缓存（全步决策时）。关闭保持严格共享步决策与更高保真输出。"}),
                "te_speed_verbose": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "关闭",
                                        "tooltip": "打印每个预测决策与跳过统计。"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "diagnostics")
    FUNCTION = "run"
    CATEGORY = "BSAI-MOSS"

    def run(self, unet, text_encoder, vae, lora_name, lora_strength, prompt, layout_json,
            steps, width, height, color_map=None, depth_map=None, control_strength=1.0,
            control_start_percent=0.0, control_end_percent=1.0, rope_enabled=False,
            scale_x=1.0, shift_x=0.0, scale_y=1.0, shift_y=0.0, shift_t=0.0,
            te_speed_enabled=False, te_speed_attention="default", te_speed_step_cache="te_predictor 2",
            te_speed_reuse_threshold=0.12, te_speed_start_percent=0.20, te_speed_end_percent=0.80,
            te_speed_predictor_error_limit=0.06, te_speed_uncond_cache=False, te_speed_verbose=False):
        lines = []
        torch.manual_seed(42)
        dev = mm.get_torch_device()

        unet_path = folder_paths.get_full_path("diffusion_models", unet)
        te_path = folder_paths.get_full_path("text_encoders", text_encoder)
        vae_path = folder_paths.get_full_path("vae", vae)
        for p, name in ((unet_path, "UNET"), (te_path, "TE"), (vae_path, "VAE")):
            if not p or not os.path.exists(p):
                raise RuntimeError("%s 文件不存在: %s" % (name, p))

        # ---- Optional layout JSON validation + compile ----
        # Appends a compact MOSS-Image layout line so the quantized UNET is
        # exercised with a layout-shaped input. If validation fails, the
        # layout is skipped (raw prompt is used) and the reason is recorded
        # in diagnostics.
        prompt_for_encode = prompt
        layout_json_clean = (layout_json or "").strip()
        if layout_json_clean and _compile_layout is not None:
            try:
                import json as _json
                _draft = _json.loads(layout_json_clean)
                _prose, _compiled, _lw, _lh = _compile_layout(_draft)
                prompt_for_encode = (
                    prompt + "\n\n" + _compiled
                    if prompt else _compiled
                )
                _l_n_items = len(_draft.get("items", []))
                lines.append(
                    "LAYOUT ok | %dx%d | %d items | compiled=%d chars"
                    % (_lw, _lh, _l_n_items, len(_compiled))
                )
                # If the layout declares canvas dimensions, surface a hint
                # to use them as EmptyLatentImage width/height in workflows.
                if _lw and _lh and (_lw != width or _lh != height):
                    lines.append(
                        "LAYOUT hint: workflow latent size should match "
                        "%dx%d (current override = %dx%d)" % (_lw, _lh, width, height)
                    )
            except Exception as _e:
                lines.append("LAYOUT skipped: %s: %s" % (type(_e).__name__, _e))
        elif layout_json_clean and _compile_layout is None:
            lines.append(
                "LAYOUT skipped: layout_validate not available "
                "(should be bundled with this plugin)."
            )

        # ---- 量化判定（不加载进 GPU，仅读 state_dict 键与 dtype）----
        u_sd, u_meta = utils.load_torch_file(unet_path, return_metadata=True)
        lines.append("UNET %s | quant=%s" % (os.path.basename(unet), _detect_quant(u_sd, "unet")[0]))
        t_sd, t_meta = utils.load_torch_file(te_path, return_metadata=True)
        lines.append("TE   %s | quant=%s" % (os.path.basename(text_encoder), _detect_quant(t_sd, "te")[0]))

        # ---- 官方加载链路（detect_layer_quantization 自动选择混合精度 ops）----
        # ComfyUI v0.42: load_diffusion_model 接收文件路径，内部自行加载 state_dict
        patcher = sd.load_diffusion_model(unet_path, model_options={})
        print("[PROBE] UNET loaded (%s)" % os.path.basename(unet), flush=True)

        # ---- 加速 LoRA（可选，作用于 UNET patcher）----
        if lora_name and lora_name != "（无）":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path or not os.path.exists(lora_path):
                raise RuntimeError("LoRA 文件不存在: %s" % lora_path)
            lora_sd, lora_meta = utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
            patcher, _ = sd.load_lora_for_models(patcher, None, lora_sd, lora_strength, 0.0,
                                                 lora_metadata=lora_meta)
            lines.append("LORA %s | strength=%.2f" % (os.path.basename(lora_name), lora_strength))
            print("[PROBE] LoRA applied: %s (strength=%.2f)" % (lora_name, lora_strength), flush=True)
        else:
            lines.append("LORA （无，未加载加速 LoRA）")

        # ComfyUI v0.42: load_clip 的 ckpt_paths 接收文件路径列表
        clip = sd.load_clip(ckpt_paths=[te_path], clip_type=CLIPType.MOSS_IMAGE,
                            embedding_directory=folder_paths.get_folder_paths("embeddings"))
        clip.load_device = dev
        clip.offload_device = mm.unet_offload_device()
        tokens = clip.tokenize(prompt_for_encode)
        cond = clip.encode_from_tokens_scheduled(tokens)
        print("[PROBE] TE encoded", flush=True)

        v_sd, _ = utils.load_torch_file(vae_path, return_metadata=True)
        # 官方 MOSS-Image VAE 为 AutoencoderKLFlux2（ComfyUI 识别为带 batch_norm_latent 的 AutoencoderKL）
        vae_model = VAE(sd=v_sd, dtype=torch.float32)
        print("[PROBE] VAE loaded", flush=True)

        # ---- VAE 校验：必须用 MOSS-Image 官方 VAE（decoder入口=32ch + 128通道BN统计）----
        # （Krea2-HD=16ch、BSAI-Qwen-Image-2.1-VAE=64ch 均无法解码 128ch MOSS latent）
        pqc = v_sd.get("post_quant_conv.weight")
        bn_mean = v_sd.get("bn.running_mean")
        vae_in = int(pqc.shape[1]) if pqc is not None else None
        bn_ch = int(bn_mean.shape[0]) if bn_mean is not None else None
        lines.append("VAE  %s | decoder入口=%s ch, BN=%s ch" % (os.path.basename(vae), vae_in, bn_ch))
        if vae_in != 32 or bn_ch != 128:
            raise RuntimeError(
                "VAE 与 MOSS-Image 不匹配：需要官方 AutoencoderKLFlux2（decoder入口=32ch、BN=128ch，"
                "即 models/vae/MOSS-Image-VAE.safetensors）；当前 %s 为 decoder入口=%s、BN=%s。" % (
                    os.path.basename(vae), vae_in, bn_ch))

        # ---- 可选 RoPE 缩放/平移探测（核心 #17172）----
        if rope_enabled:
            patcher = patcher.clone()
            patcher.set_model_rope_options(scale_x, shift_x, scale_y, shift_y, 1.0, shift_t)
            lines.append("ROPE scale_x=%.2f shift_x=%.1f scale_y=%.2f shift_y=%.1f shift_t=%.1f"
                         % (scale_x, shift_x, scale_y, shift_y, shift_t))
            print("[PROBE] rope options applied", flush=True)

        # ---- 可选色彩/深度控制探测（核心 #17206，复用已加载 MOSS VAE）----
        if color_map is not None or depth_map is not None:
            latents = [vae_model.encode(m[:, :, :, :3]) if m is not None else None
                       for m in (color_map, depth_map)]
            ref = next((lt for lt in latents if lt is not None), None)
            if ref is not None:
                batch_n = max(lt.shape[0] for lt in latents if lt is not None)
                latents = [torch.zeros_like(ref) if lt is None
                           else lt if lt.shape[-2:] == ref.shape[-2:]
                           else utils.common_upscale(lt, ref.shape[-1], ref.shape[-2],
                                                     "bilinear", "center")
                           for lt in latents]
                control = torch.cat([utils.resize_to_batch_size(lt, batch_n)
                                     for lt in latents], dim=1) * control_strength
                import node_helpers as _nh
                cond = _nh.conditioning_set_values_with_timestep_range(
                    cond, {"concat_latent_image": control},
                    control_start_percent, control_end_percent)
                kinds = ("color" if color_map is not None else "") + \
                        ("+depth" if depth_map is not None else "")
                lines.append("CONTROL %s | strength=%.2f 范围 %.3f-%.3f | latent=%s"
                             % (kinds, control_strength, control_start_percent,
                                control_end_percent, tuple(control.shape)))
                print("[PROBE] control maps applied (%s)" % kinds, flush=True)
            else:
                lines.append("CONTROL skipped: 控制图编码失败")

        # ---- 可选内置 TE-Speed MOSS 加速（模型级步预测缓存，50步场景）----
        # 在 LoRA/ROPE/控制之后、采样之前包装模型；未安装插件时优雅跳过。
        if te_speed_enabled:
            if _tespeed_cls is None:
                lines.append("TE-SPEED skipped: TE-Speed-MOSS 插件未安装（custom_nodes/TE-Speed-MOSS）")
            else:
                try:
                    _node = _tespeed_cls()
                    _kw = dict(
                        model=patcher, attention=te_speed_attention,
                        step_cache=te_speed_step_cache,
                        reuse_threshold=te_speed_reuse_threshold,
                        start_percent=te_speed_start_percent,
                        end_percent=te_speed_end_percent,
                        predictor_error_limit=te_speed_predictor_error_limit,
                        uncond_cache=te_speed_uncond_cache,
                        verbose=te_speed_verbose,
                    )
                    _res = getattr(_node, _tespeed_cls.FUNCTION)(**_kw)
                    patcher = _res[0]
                    _status = _res[1] if len(_res) > 1 else ""
                    lines.append("TE-SPEED ON | cache=%s attn=%s | %s"
                                 % (te_speed_step_cache, te_speed_attention, _status))
                    print("[PROBE] TE-Speed applied: %s" % _status, flush=True)
                except Exception as _e:
                    lines.append("TE-SPEED failed: %s: %s" % (type(_e).__name__, _e))
                    print("[PROBE] TE-Speed error: %s: %s" % (type(_e).__name__, _e), flush=True)
        else:
            lines.append("TE-SPEED off")

        # MOSS-Image 官方 latent 规格：128 通道 @ H//16（FLUX.2 式 128ch latent）
        latent = torch.randn(1, 128, height // 16, width // 16, device=dev, dtype=torch.bfloat16)
        noise = csample.prepare_noise(latent, 42, None)
        out = csample.sample(patcher, noise, steps, 1.0, "er_sde", "simple", cond, [],
                             latent_image=latent, denoise=1.0, seed=42)
        stats = "latent mean=%.4f std=%.4f nan=%d" % (
            out.mean().item(), out.std().item(), int(torch.isnan(out).float().sum().item()))
        lines.append("SAMPLE %s" % stats)
        print("[PROBE] %s" % stats, flush=True)

        # ComfyUI 的 AutoencoderKL(Flux2/batch_norm_latent) 在 decode 内部自动完成
        # BN 反归一化 + 128→32 通道 2x2 像素重排，这里直接喂 128ch @ H//16 即可
        decoded = vae_model.decode(out)  # (B, H, W, C) float [0,1]
        img = torch.clamp(decoded, 0.0, 1.0)
        img_np = (img[0].cpu().numpy() * 255.0).astype(np.uint8)  # (H, W, C)

        # ComfyUI v0.42 已移除 comfy.utils.save_image，改用 folder_paths + PIL 保存
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            "moss-quant-probe", folder_paths.get_output_directory(),
            img_np.shape[1], img_np.shape[0])
        file = "%s_%05d_.png" % (filename, counter)
        Image.fromarray(img_np).save(os.path.join(full_output_folder, file))
        png = os.path.join(full_output_folder, file)
        lines.append("saved: %s" % png)
        print("[PROBE] saved:", png, flush=True)

        batch = img.float()  # (B, H, W, C) float [0,1]
        return (batch, "\n".join(lines))


NODE_CLASS_MAPPINGS = {"BSAMOSSInt8Probe": BSAMOSSInt8Probe}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAMOSSInt8Probe": "BSA MOSS Quant Probe (INT8/FP8/NVFP4/MXFP8)"}
