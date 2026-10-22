# BSAI-MOSS-ComfyUI

[中文说明](#中文说明) | [English](#english)

---

## English

ComfyUI companion plugin for the [OpenMOSS MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)
diffusion model (5B single-stream DiT, day-0 release 2026-10-19, technical report
[arXiv:2610.23374](https://arxiv.org/abs/2610.23374)).

This plugin is **not a replacement** for the official `moss-image` ComfyUI plugin
(which ships the Loaders). It layers five BSAI-native nodes on top:

| Node | Purpose |
|------|---------|
| `BSAI_MOSS_LayoutPromptEnhancer` | Layout-aware prompt enhancer. Accepts a rough idea + an optional Layout JSON draft, dispatches to your local LLM (Strata / Nemotron / NInfer / rule_only / none), and returns a MOSS-Image-ready positive prompt (compiled JSON line for layouts, prose for captions). |
| `BSAI_MOSS_SamplerPresets` | Encodes the MOSS-Image technical-report sampling defaults (Euler / flow / 50 steps / CFG 3.0 / shift 5) as a single combo. Includes presets for the SFT-4steps-turbo LoRA, long-text rendering, and low-VRAM. |
| `MOSSImageDMDApply` | Applies a DMD main-acceleration LoRA (e.g. `4steps-turbo`) to a `MODEL`, equivalent to `LoraLoaderModelOnly`. |
| `MOSSImageDMDDataspaceSampler` | DMD few-step distilled sampler with built-in `time_embedding` + `dataspace` dual-LoRA application (Euler / simple / CFG 1.0, matching the V5 workflow 8-step DMD branch). Outputs `LATENT` with the MOSS-Image spec (128 channels @ H//16). |
| `BSAMOSSInt8Probe` | Multi-quantisation end-to-end probe (INT8 / FP8 / NVFP4 / MXFP8): pick quantised UNET + TE + VAE, run through the official ComfyUI v0.42 load chain, output diagnostics string + IMAGE. |

> The two DMD nodes were merged in from the former standalone `BSAI-MOSS-DMD`
> plugin, and the probe node from `BSAI-MOSS-INT8-Probe`. Node class names are
> unchanged, so existing workflows load as-is.

### Install

Drop the `BSAI-MOSS-ComfyUI` folder under `ComfyUI/custom_nodes/` and restart
ComfyUI. No additional Python dependencies; the plugin reuses the existing
BSAI LLM clients that ship with `BSAI-FLUX3-ComfyUI` and `BSAI-LLM-Strata`.

### Layout schema

`BSAI_MOSS_LayoutPromptEnhancer` validates Layout drafts against the official
schema documented at
`moss-image/skills/moss-image-prompt-enhancer/references/layout.md`:

```json
{
  "width": 1024,
  "height": 1024,
  "bbox_coord_space": "abs",
  "bbox_format": "xyxy",
  "summary": {
    "caption": "Global treatment prose.",
    "canvas": "single"
  },
  "items": [
    {
      "label": "object",
      "caption": "Visible appearance of the object.",
      "bbox": [176, 224, 800, 864]
    },
    {
      "label": "text",
      "caption": "Typography, colour, placement treatment.",
      "text": "The literal visible string.",
      "bbox": [928, 272, 1408, 448]
    }
  ]
}
```

The `layout_json` output is the compiled form (`abs_centralized` / `yxyx`
integer boxes, no canvas metadata) ready to paste into MOSS-Image's positive
prompt. Width/height extracted from the draft are returned on dedicated INT
outputs so they can be wired into `EmptyLatentImage` without manual entry.

### Engines

`engine` accepts the same set as `BSAI_FLUX3_PromptEnhancer`:

| Engine | Behaviour |
|--------|-----------|
| `auto` | Try NInfer → Strata → Nemotron (in that order). |
| `strata_local` | Only the local Strata server (127.0.0.1:8080). |
| `nemotron` | Only the local Nemotron llama.cpp server (127.0.0.1:8085). |
| `ninfer` | Only the bundled `ninfer-serve.exe` (127.0.0.1:8082). |
| `rule_only` | Skip the LLM entirely; return the user input verbatim. |
| `none` | Pure passthrough; raw_prompt + layout_json are validated and passed through unmodified. |

---

## 中文说明

[OpenMOSS MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) 扩散模型
（5B 单流 DiT，2026-10-19 day-0 发布，技术报告
[arXiv:2610.23374](https://arxiv.org/abs/2610.23374)）的 ComfyUI 配套插件。

本插件**不是**官方 `moss-image` 插件（加载器由官方插件提供）的替代品，
而是在其上叠加的五个 BSAI 自研节点：

| 节点 | 作用 |
|------|------|
| `BSAI_MOSS_LayoutPromptEnhancer` | 布局感知提示词增强器。输入粗略想法 + 可选 Layout JSON 草稿，分发到本地 LLM（Strata / Nemotron / NInfer / rule_only / none），输出 MOSS-Image 可直接使用的正面提示词（布局场景输出编译后的 JSON 行，普通场景输出散文描述）。 |
| `BSAI_MOSS_SamplerPresets` | 把技术报告推荐采样参数（Euler / flow / 50 步 / CFG 3.0 / shift 5）封装为单个下拉选项，另含 SFT-4steps-turbo LoRA、长文本渲染、低显存三套预设。 |
| `MOSSImageDMDApply` | 对 `MODEL` 应用 DMD 主加速 LoRA（如 `4steps-turbo`），等价于 `LoraLoaderModelOnly`。 |
| `MOSSImageDMDDataspaceSampler` | DMD 少步蒸馏采样器，内置 `time_embedding` + `dataspace` 双 LoRA 应用（euler / simple / CFG 1.0，与 V5 工作流 8 步 DMD 分支一致），输出符合 MOSS-Image 规格的 `LATENT`（128 通道 @ H//16）。 |
| `BSAMOSSInt8Probe` | 多量化格式全链路探针（INT8 / FP8 / NVFP4 / MXFP8）：选择量化 UNET + TE + VAE，走官方 v0.42 加载链路，输出诊断字符串 + IMAGE。 |

> 两个 DMD 节点已从原独立插件 `BSAI-MOSS-DMD` 并入本插件，
> 探针节点已从原独立插件 `BSAI-MOSS-INT8-Probe` 并入本插件。
> 节点类名均保持不变，已有工作流可直接加载，无需修改。

### 安装

把 `BSAI-MOSS-ComfyUI` 文件夹放入 `ComfyUI/custom_nodes/` 并重启 ComfyUI。
无需额外 Python 依赖；插件复用 `BSAI-FLUX3-ComfyUI` 与 `BSAI-LLM-Strata`
自带的 BSAI LLM 客户端。

### 布局 schema

`BSAI_MOSS_LayoutPromptEnhancer` 按官方 schema
（`moss-image/skills/moss-image-prompt-enhancer/references/layout.md`）
校验 Layout 草稿，示例见上方英文部分 JSON。

`layout_json` 输出为编译后形式（`abs_centralized` / `yxyx` 整数坐标框，
不含画布元数据），可直接拼入 MOSS-Image 正面提示词。草稿中的宽高会通过
独立的 INT 输出端口返回，可直接连到 `EmptyLatentImage`，无需手动填写。

### 引擎

`engine` 选项与 `BSAI_FLUX3_PromptEnhancer` 一致：

| 引擎 | 行为 |
|------|------|
| `auto` | 依次尝试 NInfer → Strata → Nemotron。 |
| `strata_local` | 仅本地 Strata 服务（127.0.0.1:8080）。 |
| `nemotron` | 仅本地 Nemotron llama.cpp 服务（127.0.0.1:8085）。 |
| `ninfer` | 仅内置 `ninfer-serve.exe`（127.0.0.1:8082）。 |
| `rule_only` | 完全跳过 LLM，原样返回用户输入。 |
| `none` | 纯直通；raw_prompt + layout_json 校验后原样输出。 |
