"""SRVGGNetCompact neural network for Real-ESRGAN super resolution."""

import torch.nn as nn
import torch.nn.functional as F

from config import SR_NUM_IN_CH, SR_NUM_OUT_CH, SR_NUM_FEAT, SR_NUM_CONV, SR_UPSCALE


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style SR network matching realesr-general-x4v3.pth."""

    def __init__(self):
        super().__init__()
        self.upscale = SR_UPSCALE

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(SR_NUM_IN_CH, SR_NUM_FEAT, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=SR_NUM_FEAT))
        for _ in range(SR_NUM_CONV):
            self.body.append(nn.Conv2d(SR_NUM_FEAT, SR_NUM_FEAT, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=SR_NUM_FEAT))
        self.body.append(nn.Conv2d(SR_NUM_FEAT, SR_NUM_OUT_CH * (SR_UPSCALE ** 2), 3, 1, 1))
        self.upsampler = nn.PixelShuffle(SR_UPSCALE)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        return out + base
