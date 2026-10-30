# v14.234: MOSSMergePreset —— 核心 ModelMergeMOSSImage 的 BSAI 包装。
# 加 preset 下拉(Balanced 等) + strength 全局倍率, 内部走核心
# ModelMergeBlocks.merge 的前缀匹配逻辑, 工作流 JSON 里的 m_xxx 参数名
# 直接映射到核心 key 前缀。
#
# Balanced preset 值(截图实测):
#   video_embedder=0.20  text_proj=0.20  text_blocks=0.16/0.16
#   time_embedding=0.24  shared_modulation=0.20  layers.0~41=0.16

# 核心 key 前缀表(与 comfy_extras/nodes_model_merging_model_specific.py
# ModelMergeMOSSImage.INPUT_TYPES 完全一致)。
_KEY_PREFIXES = [
    "video_embedder.",
    "text_embedder.proj.",
    "text_embedder.blocks.0.",
    "text_embedder.blocks.1.",
    "time_embedding.",
    "shared_block_modulation.",
]
for _i in range(42):
    _KEY_PREFIXES.append("layers.{}.".format(_i))
_KEY_PREFIXES += [
    "final_norm_video.",
    "final_linear_video.",
    "video_head_modulation.",
]

# preset 表: preset_name -> {key_prefix: ratio}
_PRESETS = {
    "Balanced": {
        "video_embedder.": 0.20,
        "text_embedder.proj.": 0.20,
        "text_embedder.blocks.0.": 0.16,
        "text_embedder.blocks.1.": 0.16,
        "time_embedding.": 0.24,
        "shared_block_modulation.": 0.20,
    },
    "SFT-heavy": {
        "video_embedder.": 0.30,
        "text_embedder.proj.": 0.30,
        "text_embedder.blocks.0.": 0.25,
        "text_embedder.blocks.1.": 0.25,
        "time_embedding.": 0.35,
        "shared_block_modulation.": 0.30,
    },
    "Base-heavy": {
        "video_embedder.": 0.10,
        "text_embedder.proj.": 0.10,
        "text_embedder.blocks.0.": 0.08,
        "text_embedder.blocks.1.": 0.08,
        "time_embedding.": 0.12,
        "shared_block_modulation.": 0.10,
    },
}
_DEFAULT_LAYER_RATIO = 0.16  # layers.0~41 默认


class MOSSMergePreset:
    """BSAI 包装: base_model + sft_model 按 preset 合并为 MOSS-Image 权重。"""

    @classmethod
    def INPUT_TYPES(s):
        float_arg = ("FLOAT", {"default": 0.16, "min": 0.0, "max": 1.0, "step": 0.01})
        req = {
            "base_model": ("MODEL",),
            "sft_model": ("MODEL",),
            "preset": (["Balanced", "SFT-heavy", "Base-heavy", "Custom"],),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
        # 手动覆盖参数(Custom preset 时生效; 选 Balanced 等时自动填 preset 值但可微调)
        req["m_video_embedder"] = float_arg
        req["m_text_proj"] = float_arg
        req["m_text_block0"] = float_arg
        req["m_text_block1"] = float_arg
        req["m_time"] = float_arg
        req["m_shared"] = float_arg
        for i in range(42):
            req["m_L{:02d}".format(i)] = float_arg
        return {"required": req}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "merge"
    CATEGORY = "BSAI/MOSS"

    def merge(self, base_model, sft_model, preset, strength,
              m_video_embedder, m_text_proj, m_text_block0, m_text_block1,
              m_time, m_shared, **kwargs):
        # 构造核心 key 前缀 -> ratio 表
        if preset in _PRESETS:
            p = _PRESETS[preset]
            ratios = {prefix: p.get(prefix, _DEFAULT_LAYER_RATIO) for prefix in _KEY_PREFIXES}
            # preset 值覆盖 m_xxx 手动参数
            ratios["video_embedder."] = p.get("video_embedder.", m_video_embedder)
            ratios["text_embedder.proj."] = p.get("text_embedder.proj.", m_text_proj)
            ratios["text_embedder.blocks.0."] = p.get("text_embedder.blocks.0.", m_text_block0)
            ratios["text_embedder.blocks.1."] = p.get("text_embedder.blocks.1.", m_text_block1)
            ratios["time_embedding."] = p.get("time_embedding.", m_time)
            ratios["shared_block_modulation."] = p.get("shared_block_modulation.", m_shared)
        else:
            # Custom: 全用 m_xxx 手动值
            ratios = {
                "video_embedder.": m_video_embedder,
                "text_embedder.proj.": m_text_proj,
                "text_embedder.blocks.0.": m_text_block0,
                "text_embedder.blocks.1.": m_text_block1,
                "time_embedding.": m_time,
                "shared_block_modulation.": m_shared,
            }
            for i in range(42):
                ratios["layers.{}.".format(i)] = kwargs.get("m_L{:02d}".format(i), _DEFAULT_LAYER_RATIO)

        # strength 全局倍率
        if strength != 1.0:
            ratios = {k: v * strength for k, v in ratios.items()}

        # 走核心 ModelMergeBlocks.merge 的前缀匹配逻辑
        m = base_model.clone()
        kp = sft_model.get_key_patches("diffusion_model.")
        for k in kp:
            ratio = _DEFAULT_LAYER_RATIO
            k_unet = k[len("diffusion_model."):]
            last_arg_size = 0
            for prefix in ratios:
                if k_unet.startswith(prefix) and last_arg_size < len(prefix):
                    ratio = ratios[prefix]
                    last_arg_size = len(prefix)
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)

        print(f"[BSAI MOSSMergePreset] preset={preset} strength={strength} "
              f"video_embedder={ratios['video_embedder.']:.2f} "
              f"text_proj={ratios['text_embedder.proj.']:.2f} "
              f"time={ratios['time_embedding.']:.2f} "
              f"shared={ratios['shared_block_modulation.']:.2f}")
        return (m,)


NODE_CLASS_MAPPINGS = {"MOSSMergePreset": MOSSMergePreset}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSMergePreset": "BSAI MOSS Merge Preset"}
