import torch
import torch.nn.functional as F
from torch import nn

from ..conv import Conv, DWConv


class PSICConv(Conv):
    """
    Persistent Structure and Illumination Calibration Convolution.

    Designed for robust traffic-sign feature extraction under
    illumination variation and adverse-weather interference.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p=None,
        g: int = 1,
        d: int = 1,
        act=True,
    ):
        # Keep the original Conv parameter names:
        # self.conv / self.bn / self.act
        super().__init__(c1, c2, k, s, p, g, d, act)

        # Fuse:
        # stable structure
        # disturbance cue
        # illumination variation
        self.reduce = Conv(c2 * 3, c2, 1, 1)

        # Lightweight local refinement
        self.mix = DWConv(c2, c2, 3, 1)

        # Residual correction
        self.out = nn.Conv2d(
            c2,
            c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Identity-preserving initialization
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _calibrate(self, y: torch.Tensor) -> torch.Tensor:
        # Local responses at two neighboring spatial scales
        mean3 = F.avg_pool2d(
            y,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )

        mean5 = F.avg_pool2d(
            y,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )

        # Local residuals
        r3 = y - mean3
        r5 = y - mean5

        # Preserve the more conservative response shared
        # by the two spatial scales.
        persistent = torch.where(
            r3.abs() <= r5.abs(),
            r3,
            r5,
        )

        # Stable local structure
        structure = mean3 + persistent

        # Scale-inconsistent component
        disturbance = y - structure

        # Local low-frequency / illumination variation
        illumination = (mean3 - mean5).abs()

        # No manually assigned fusion coefficients
        descriptor = torch.cat(
            (structure, disturbance, illumination),
            dim=1,
        )

        correction = self.reduce(descriptor)
        correction = self.mix(correction)
        correction = self.out(correction)

        return y + correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn(self.conv(x)))
        return self._calibrate(y)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv(x))
        return self._calibrate(y)