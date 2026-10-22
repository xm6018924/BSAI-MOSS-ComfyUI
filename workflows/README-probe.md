# BSAI-MOSS-INT8-Probe 示例工作流

## 工作流列表

### 1. BSAI_MOSS_INT8_Probe_快速验证流水线.json

**用途**：一键验证 INT8 量化的 MOSS-Image 模型是否能正常跑通完整推理流水线。

**包含节点**：
- `BSA MOSS INT8 Probe` — 核心探针节点（独立运行，无输入）
- `PreviewImage` — 输出图像预览
- `Note` — 插件优势与使用说明

**插件核心优势**：

| 优势 | 说明 |
|------|------|
| 单节点全链路验证 | 一个节点跑通：UNet 加载 → 文本编码 → VAE 加载 → 采样 → 解码 |
| 零配置开箱即用 | 硬编码标准路径，无需手动配置模型路径 |
| 快速定位问题 | 控制台分阶段输出日志，快速定位故障环节 |
| INT8 量化专用 | 专门用于验证 INT8 转换后的 MOSS-Image 模型 |

**使用方法**：
1. 确保 INT8 模型权重放在对应路径（见主 README）
2. 在 ComfyUI 中加载此工作流
3. 直接点击 Queue Prompt 运行
4. 查看控制台日志确认各阶段状态
5. 在预览窗口查看生成的测试图像

---

## English

## Example Workflows

### 1. BSAI_MOSS_INT8_Probe_快速验证流水线.json

**Purpose**: One-click verification that the INT8 quantized MOSS-Image model can run the full inference pipeline end-to-end.

**Key Advantages**:

| Advantage | Description |
|-----------|-------------|
| Single-node full pipeline | One node covers: UNet load → text encode → VAE load → sampling → decode |
| Zero config | Hardcoded standard paths, no manual setup needed |
| Fast debugging | Staged console logs to quickly locate failure points |
| INT8-specific | Purpose-built for verifying INT8-converted MOSS-Image models |
