import torch
import torch.nn.functional as F
from torch import nn

from ..conv import Conv, DWConv


class EFCM(nn.Module):
    """Environmental Feature Calibration Module for adverse traffic scenes."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.proj = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.illumination = DWConv(c2, c2, 3)
        self.interference = DWConv(c2, c2, 3)
        self.fuse = Conv(c2 * 2, c2, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)

        local_mean = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
        centered = x - local_mean
        local_variance = F.avg_pool2d(centered.square(), 3, 1, 1, count_include_pad=False)

        illumination = self.illumination(centered)
        interference = self.interference(local_variance)

        calibration = self.fuse(torch.cat((illumination, interference), dim=1))
        return x + calibration