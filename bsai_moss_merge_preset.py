# v14.235: MOSSMergePreset —— 核心 ModelMergeMOSSImage 的 BSAI 包装。
# 加 preset 下拉(Balanced 等) + strength 全局倍率, 内部走核心
# ModelMergeBlocks.merge 的前缀匹配逻辑, 工作流 JSON 里的 m_xxx 参数名
# 直接映射到核心 key 前缀。
#
# Balanced preset 值(截图实测):
#   video_embedder=0.20  text_proj=0.20  text_blocks=0.16/0.16
#   time_embedding=0.24  shared_modulation=0.20  layers.0~41=0.16
#
# v14.235 修复(2026-11-16): 用 INT8/FP8/NVFP4 量化模型合并时报
#   AttributeError: 'Linear' object has no attribute 'weight_scale'
#   位置: comfy/model_patcher.py get_key_patches -> get_key_weight
# 根因: 旧实现直接调用 sft_model.get_key_patches("diffusion_model.")。
#   comfy 的 get_key_patches 遍历的是 state_dict, 并对每个键做 getattr(模块, 属性)。
#   量化权重在加载时被 mixed_precision_ops 打包进 QuantizedTensor, 模块上只保留
#   weight/bias, 而 state_dict 仍会序列化出 *.weight_scale、*.comfy_quant、
#   *.input_scale 等辅助键 —— 这些键没有对应的模块属性, 于是 getattr 直接抛
#   AttributeError。只要 sft_model 是量化权重, 合并必崩, 与采样无关。
# 修复: 改用 named_parameters 遍历模块真实参数(weight/bias), 量化权重连同
#   模块自身的 convert_func(反量化)一起交给 comfy 的 patch 机制, 由 comfy 在
#   真正取权重时再反量化, 合并阶段不展开权重, 内存占用与原生合并一致。
#   该写法与 ComfyUI-KJNodes 对同一问题的处理方式相同。

import torch

import comfy.model_patcher
import comfy.utils

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

_DIFFUSION_PREFIX = "diffusion_model."
_MERGE_SUFFIXES = (".weight", ".bias")


def _identity(tensor, **kwargs):
    return tensor


def _key_patches(model):
    """量化安全的 {state-dict key: patch 列表}。

    等价于 comfy 的 ModelPatcher.get_key_patches(), 区别是遍历 named_parameters
    而不是 state_dict: 只处理真实参数 weight/bias, 自动跳过 *.weight_scale /
    *.comfy_quant / *.input_scale 等量化序列化辅助键。

    量化权重不做提前反量化: 连同模块的 convert_func 一起放进 patch, comfy 在
    实际取权重时才调用 convert_func 展开成浮点, 因此合并阶段的内存占用和原生
    ModelMergeBlocks 一致(不会把 5G 的 INT8 权重展开成 10G BF16)。
    """
    p = {}
    with model.use_ejected():
        for k, _ in model.model.named_parameters():
            if not k.startswith(_DIFFUSION_PREFIX) or not k.endswith(_MERGE_SUFFIXES):
                continue

            weight, _, convert_func = comfy.model_patcher.get_key_weight(model.model, k)
            if not isinstance(weight, torch.Tensor):
                continue

            bk = model.backup.get(k, None)
            if bk is not None:
                weight = bk.weight
            hbk = model.hook_backup.get(k, None)
            if hbk is not None:
                weight = hbk[0]
            if convert_func is None:
                convert_func = _identity

            p[k] = [(weight, convert_func)] + list(model.patches.get(k, []))
    return p


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
        kp = _key_patches(sft_model)
        merged = 0
        for k in kp:
            ratio = _DEFAULT_LAYER_RATIO
            k_unet = k[len(_DIFFUSION_PREFIX):]
            last_arg_size = 0
            for prefix in ratios:
                if k_unet.startswith(prefix) and last_arg_size < len(prefix):
                    ratio = ratios[prefix]
                    last_arg_size = len(prefix)
            if m.add_patches({k: kp[k]}, 1.0 - ratio, ratio):
                merged += 1

        if merged == 0:
            raise ValueError(
                "[BSAI MOSSMergePreset] base_model 与 sft_model 没有可合并的公共权重, "
                "请确认两个模型都是 MOSS-Image 权重。")

        print(f"[BSAI MOSSMergePreset] preset={preset} strength={strength} "
              f"video_embedder={ratios['video_embedder.']:.2f} "
              f"text_proj={ratios['text_embedder.proj.']:.2f} "
              f"time={ratios['time_embedding.']:.2f} "
              f"shared={ratios['shared_block_modulation.']:.2f} "
              f"merged_params={merged}")
        return (m,)


NODE_CLASS_MAPPINGS = {"MOSSMergePreset": MOSSMergePreset}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSMergePreset": "BSAI MOSS Merge Preset"}
