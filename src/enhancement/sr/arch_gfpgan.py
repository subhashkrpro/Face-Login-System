"""
GFPGANv1Clean — face restoration architecture (inference only).

Minimal reimplementation of GFPGANv1Clean from the official GFPGAN repo
(TencentARC/GFPGAN).  Weights are binary-compatible with the official
GFPGANv1.4.pth checkpoint.

Architecture:
    - Degradation-aware U-Net encoder/decoder
    - Pretrained StyleGAN2 decoder (clean, no custom CUDA ops)
    - Spatial Feature Transform (SFT) modulation

Reference:
    Wang et al., "GFP-GAN: Towards Real-World Blind Face Restoration
    with Generative Facial Prior", CVPR 2021.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_stylegan2 import StyleGAN2GeneratorClean


# ── Encoder building blocks ──────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual block with bilinear down/up-sampling.

    Uses plain ``nn.Conv2d`` (no equalized lr) — matches the GFPGAN v1.4
    checkpoint exactly.
    """

    def __init__(self, in_ch: int, out_ch: int, mode: str = "down"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.scale_factor = 0.5 if mode == "down" else 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        out = F.interpolate(out, scale_factor=self.scale_factor,
                            mode="bilinear", align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        skip = F.interpolate(x, scale_factor=self.scale_factor,
                             mode="bilinear", align_corners=False)
        skip = self.skip(skip)
        return (out + skip) / math.sqrt(2)


# ── GFPGANv1Clean ────────────────────────────────────────────────────────

class GFPGANv1Clean(nn.Module):
    """
    GFP-GAN v1 Clean architecture (no custom CUDA ops).

    Args:
        out_size: Output spatial size (must be 512 for v1.4).
        num_style_feat: Style vector dimensionality (512).
        channel_multiplier: Channel multiplier for StyleGAN2 decoder.
        narrow: Channel narrowing factor (0.5 for GFPGANv1.4).
        sft_half: If True, only apply SFT to the first half of channels.
        num_mlp: Number of MLP layers in StyleGAN2 mapping network.
        input_is_latent: Whether the input to the decoder is already a
            latent code (``True`` for GFPGAN).
        different_w: Use per-layer W+ style codes.
    """

    def __init__(self, out_size: int = 512, num_style_feat: int = 512,
                 channel_multiplier: int = 2, narrow: float = 1,
                 sft_half: bool = True, num_mlp: int = 8,
                 input_is_latent: bool = True, different_w: bool = True):
        super().__init__()
        self.sft_half = sft_half
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        channels = {
            "4": int(512 * narrow),
            "8": int(512 * narrow),
            "16": int(512 * narrow),
            "32": int(512 * narrow),
            "64": int(256 * channel_multiplier * narrow),
            "128": int(128 * channel_multiplier * narrow),
            "256": int(64 * channel_multiplier * narrow),
            "512": int(32 * channel_multiplier * narrow),
            "1024": int(16 * channel_multiplier * narrow),
        }

        self.log_size = int(math.log2(out_size))

        # ── U-Net encoder (downsample) ────────────────────────────────
        first_out_ch = channels[str(out_size)]
        self.conv_body_first = nn.Conv2d(3, first_out_ch, 1)

        in_ch = first_out_ch
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_ch = channels[str(2 ** (i - 1))]
            self.conv_body_down.append(ResBlock(in_ch, out_ch, mode="down"))
            in_ch = out_ch

        # Bottleneck conv (4×4 feature map)
        self.final_conv = nn.Conv2d(in_ch, channels["4"], 3, 1, 1)

        # Map bottleneck to W+ style code
        if different_w:
            linear_out = (self.log_size * 2 - 2) * num_style_feat
        else:
            linear_out = num_style_feat
        self.final_linear = nn.Linear(channels["4"] * 4 * 4, linear_out)

        # ── U-Net decoder (upsample) ──────────────────────────────────
        in_ch = channels["4"]
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_ch = channels[str(2 ** i)]
            self.conv_body_up.append(ResBlock(in_ch, out_ch, mode="up"))
            in_ch = out_ch

        # Per-level RGB heads
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[str(2 ** i)], 3, 1))

        # ── StyleGAN2 decoder ─────────────────────────────────────────
        # The pretrained decoder always uses narrow=1 (full width) even
        # when the GFPGAN encoder uses narrow<1.  sft_half bridges the
        # channel mismatch by applying conditions to only half the
        # decoder channels.
        self.stylegan_decoder = StyleGAN2GeneratorClean(
            out_size, num_style_feat=num_style_feat,
            num_mlp=num_mlp, channel_multiplier=channel_multiplier,
            narrow=1,
        )

        # ── SFT modulation layers (condition_scale + condition_shift) ─
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()

        for i in range(3, self.log_size + 1):
            out_ch = channels[str(2 ** i)]
            sft_out = out_ch  # same regardless of sft_half
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_ch, sft_out, 3, 1, 1),
                )
            )
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_ch, sft_out, 3, 1, 1),
                )
            )

    def forward(self, x: torch.Tensor,
                return_latents: bool = False,
                return_rgb: bool = True,
                randomize_noise: bool = True,
                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass.

        Returns:
            (restored_image, intermediate_rgbs)
        """
        conditions: list[torch.Tensor] = []
        unet_skips: list[torch.Tensor] = []
        out_rgbs: list[torch.Tensor] = []

        # ── Encoder ───────────────────────────────────────────────────
        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)

        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)  # prepend → reversed order

        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)

        # Map to style code (W+)
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(
                style_code.size(0), -1, self.num_style_feat)

        # ── Decoder (U-Net up + SFT conditions) ──────────────────────
        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)

            # SFT conditions
            scale = self.condition_scale[i](feat)
            conditions.append(scale.clone())
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())

            # Intermediate RGB
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        # ── StyleGAN2 decoder ─────────────────────────────────────────
        image, _ = self.stylegan_decoder(
            [style_code],
            conditions=conditions,
            input_is_latent=self.input_is_latent,
            return_latents=return_latents,
            randomize_noise=randomize_noise,
            sft_half=self.sft_half,
        )

        return image, out_rgbs
