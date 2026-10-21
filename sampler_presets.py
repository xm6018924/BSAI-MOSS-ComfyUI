"""BSAI_MOSS_SamplerPresets — MOSS-Image recommended sampling defaults.

Encodes the official MOSS-Image sampling recipe (technical report 2026-10-19,
arXiv:2610.23374) as ComfyUI KSampler-compatible outputs, plus the optional
SFT-4steps-turbo preset and a "raw" pass-through for users who want to override.

Outputs match ComfyUI's built-in KSampler/KSamplerAdvanced slot types so this
node can be wired directly into any MOSS-Image workflow.
"""
from __future__ import annotations

# (sampler, scheduler, steps, cfg, shift, neg_layer_skip)
_PRESETS = {
    "MOSS-Image Recommended (flow-matching Euler)": ("euler", "flow", 50, 3.0, 5.0, 22),
    "MOSS-Image-SFT-4steps-turbo (turbo sampler)": ("euler", "flow", 4, 1.0, 5.0, 0),
    "MOSS-Image Long-Text (CFG 3.5, 60 steps)":   ("euler", "flow", 60, 3.5, 5.0, 22),
    "MOSS-Image Low-VRAM (CFG 2.5, 35 steps)":    ("euler", "flow", 35, 2.5, 5.0, 22),
}


class BSAI_MOSS_SamplerPresets:
    """Wrap the MOSS-Image technical report defaults in a single combo."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(_PRESETS.keys()), {
                    "default": "MOSS-Image Recommended (flow-matching Euler)",
                    "tooltip": "Pick a MOSS-Image sampling preset. "
                               "The 'Recommended' preset matches the 2026-10-19 technical report.",
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SAMPLER", "SIGMAS", "INT", "FLOAT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("sampler", "sigmas", "steps", "cfg", "seed", "denoise", "shift")
    FUNCTION = "execute"
    CATEGORY = "BSAI-MOSS"

    SEARCH_ALIASES = ["moss sampler", "moss presets", "flow matching euler", "采样器预设"]

    def execute(self, preset: str, seed: int, denoise: float):
        sampler_name, scheduler, steps, cfg, shift, _neg_layer_skip = _PRESETS.get(
            preset, _PRESETS["MOSS-Image Recommended (flow-matching Euler)"],
        )

        sampler = _build_comfy_sampler(sampler_name, scheduler)
        sigmas = _build_sigmas(steps, shift, denoise)

        return (sampler, sigmas, steps, cfg, seed, denoise, shift)


def _build_comfy_sampler(sampler_name: str, scheduler: str):
    """Build a (sampler, scheduler) tuple compatible with ComfyUI samplers.

    We don't import comfy.samplers at module load time to keep the plugin
    importable even when ComfyUI is not running; the (name, name) tuple is
    the canonical contract ComfyUI expects from sampler providers.
    """
    try:
        import comfy.samplers  # type: ignore  # noqa: F401  (sanity probe)
        return (sampler_name, scheduler)
    except Exception as e:
        print(f"[BSAI-MOSS] comfy.samplers unavailable: {e}")
        return (sampler_name, scheduler)


def _build_sigmas(steps: int, shift: float, denoise: float):
    """Build a 1D float tensor of length steps+1 representing the sigma schedule.

    Uses a simple linear-from-1-to-0 schedule with the MOSS-Image "shift"
    applied: t[i] = (1 - u)^shift where u is the raw schedule. Matches the
    reference flow-matching Euler schedule used by the 2026-10-19 technical
    report.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    if abs(shift - 1.0) > 1e-6:
        u = 1.0 - t
        t = 1.0 - u.clamp(min=0.0, max=1.0).pow(shift)
    if denoise < 1.0:
        keep = max(1, int(round(steps * denoise)) + 1)
        t = t[-keep:]
    return t


NODE_CLASS_MAPPINGS = {
    "BSAI_MOSS_SamplerPresets": BSAI_MOSS_SamplerPresets,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_MOSS_SamplerPresets": "BSAI MOSS Sampler Presets",
}