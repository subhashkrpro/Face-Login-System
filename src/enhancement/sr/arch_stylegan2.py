"""
StyleGAN2 Generator (clean version — no custom CUDA ops).

Minimal inference-only implementation of the StyleGAN2 generator used as
the decoder inside GFPGANv1Clean.  Ported from the official GFPGAN repo
(TencentARC/GFPGAN) — weights are binary-compatible with GFPGANv1.4.pth.

Reference:
    Karras et al., "Analyzing and Improving the Image Quality of StyleGAN",
    CVPR 2020.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────

class NormStyleCode(nn.Module):
    """Normalize style code to unit sphere."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True,
                 bias_init_val: float = 0.0, lr_mul: float = 1.0,
                 activation: str | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = (
            nn.Parameter(torch.full([out_dim], float(bias_init_val)))
            if bias else None
        )
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation:
            out = F.linear(x, self.weight * self.scale)
            out = F.leaky_relu(out + self.bias * self.lr_mul, 0.2) * math.sqrt(2)
        else:
            out = F.linear(
                x, self.weight * self.scale,
                self.bias * self.lr_mul if self.bias is not None else None,
            )
        return out


class ModulatedConv2d(nn.Module):
    """Modulated convolution (style-based)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 num_style_feat: int, demodulate: bool = True,
                 sample_mode: str | None = None, eps: float = 1e-8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        self.scale = 1 / math.sqrt(in_ch * kernel_size ** 2)
        self.modulation = EqualLinear(num_style_feat, in_ch, bias=True,
                                      bias_init_val=1.0)
        self.weight = nn.Parameter(
            torch.randn(1, out_ch, in_ch, kernel_size, kernel_size)
        )
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # Upsample input BEFORE conv (clean/bilinear approach)
        if self.sample_mode == "upsample":
            x = F.interpolate(x, scale_factor=2, mode="bilinear",
                              align_corners=False)
        elif self.sample_mode == "downsample":
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear",
                              align_corners=False)

        _, _, h_new, w_new = x.shape

        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * self.scale * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_ch, 1, 1, 1)

        weight = weight.view(
            b * self.out_ch, c, self.kernel_size, self.kernel_size)

        x = x.view(1, b * c, h_new, w_new)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        _, _, ho, wo = out.shape
        out = out.view(b, self.out_ch, ho, wo)

        return out


class StyleConv(nn.Module):
    """Modulated conv + noise injection + leaky ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 num_style_feat: int, demodulate: bool = True,
                 sample_mode: str | None = None):
        super().__init__()
        self.modulated_conv = ModulatedConv2d(
            in_ch, out_ch, kernel_size, num_style_feat,
            demodulate=demodulate, sample_mode=sample_mode,
        )
        self.weight = nn.Parameter(torch.zeros(1))           # noise scale
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.activate = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                noise: torch.Tensor | None = None) -> torch.Tensor:
        out = self.modulated_conv(x, style)
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        out = self.activate(out + self.bias)
        return out


class ToRGB(nn.Module):
    """Maps features to RGB via 1x1 modulated conv."""

    def __init__(self, in_ch: int, num_style_feat: int, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_ch, 3, 1, num_style_feat, demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                skip: torch.Tensor | None = None) -> torch.Tensor:
        out = self.modulated_conv(x, style) + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(
                    skip, scale_factor=2, mode="bilinear", align_corners=False,
                )
            out = out + skip
        return out


class ConstantInput(nn.Module):
    """Learned constant input for the StyleGAN2 generator."""

    def __init__(self, num_ch: int, size: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_ch, size, size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight.repeat(x.shape[0], 1, 1, 1)


# ── Generator ─────────────────────────────────────────────────────────────

class StyleGAN2GeneratorClean(nn.Module):
    """
    StyleGAN2 generator (clean, no custom CUDA ops).

    Args:
        out_size: Output spatial size (e.g. 512).
        num_style_feat: Dimensionality of style vectors (default: 512).
        num_mlp: Number of MLP layers in the style mapping network.
        channel_multiplier: Channel multiplier for larger models.
        narrow: Channel narrowing factor (0.5 for GFPGANv1.4).
    """

    def __init__(self, out_size: int, num_style_feat: int = 512,
                 num_mlp: int = 8, channel_multiplier: int = 2,
                 narrow: float = 1):
        super().__init__()

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
        self.num_style_feat = num_style_feat
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        # Style mapping network: [NormStyleCode, EqualLinear] * num_mlp
        style_mlp_layers: list[nn.Module] = []
        for _ in range(num_mlp):
            style_mlp_layers.append(NormStyleCode())
            style_mlp_layers.append(
                EqualLinear(num_style_feat, num_style_feat, bias=True,
                            bias_init_val=0, lr_mul=0.01, activation="fused_lrelu")
            )
        self.style_mlp = nn.Sequential(*style_mlp_layers)

        # Constant input (4x4)
        self.constant_input = ConstantInput(channels["4"], size=4)

        # First conv
        self.style_conv1 = StyleConv(
            channels["4"], channels["4"], 3, num_style_feat, demodulate=True,
        )
        self.to_rgb1 = ToRGB(channels["4"], num_style_feat, upsample=False)

        # Upsampling layers
        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        # Noise buffers
        self.noises = nn.Module()
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise{layer_idx}", torch.randn(*shape))

        in_ch = channels["4"]
        for i in range(3, self.log_size + 1):
            out_ch = channels[str(2 ** i)]
            self.style_convs.append(
                StyleConv(in_ch, out_ch, 3, num_style_feat, demodulate=True,
                          sample_mode="upsample")
            )
            self.style_convs.append(
                StyleConv(out_ch, out_ch, 3, num_style_feat, demodulate=True)
            )
            self.to_rgbs.append(ToRGB(out_ch, num_style_feat))
            in_ch = out_ch

    def forward(self, styles: list[torch.Tensor],
                conditions: list[torch.Tensor] | None = None,
                input_is_latent: bool = False,
                randomize_noise: bool = True,
                return_latents: bool = False,
                sft_half: bool = False) -> tuple[torch.Tensor, ...]:
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]

        # Noise
        if randomize_noise:
            noise = [None] * self.num_layers
        else:
            noise = [
                getattr(self.noises, f"noise{i}") for i in range(self.num_layers)
            ]

        # Broadcast single style to all layers
        if len(styles) == 1:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        elif len(styles) == 2:
            latent = torch.cat([
                styles[0].unsqueeze(1).repeat(1, self.num_latent // 2, 1),
                styles[1].unsqueeze(1).repeat(
                    1, self.num_latent - self.num_latent // 2, 1),
            ], dim=1)
        else:
            latent = torch.stack(styles, dim=1)

        out = self.constant_input(latent)
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.style_convs[0::2], self.style_convs[1::2],
            noise[1::2], noise[2::2], self.to_rgbs,
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            # SFT conditions from GFPGAN encoder
            if conditions is not None and len(conditions) > 0:
                if sft_half:
                    out_same, out_sft = torch.split(
                        out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:
                    out = out * conditions[i - 1] + conditions[i]
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        return image, None
