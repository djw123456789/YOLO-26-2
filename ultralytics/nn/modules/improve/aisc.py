import torch
import torch.nn as nn
import torch.nn.functional as F


class AISC(nn.Module):
    """
    Affine-Invariant Semantic Calibration (AISC).

    For a local feature vector f, assume an adverse-condition nuisance:
        f' = a * f + b,  a > 0
    where b is shared across channels.

    Define:
        mu = mean_c(f)
        r  = f - mu
        s  = sqrt(mean_c(r^2) + eps)
        u  = r / s

    Then u(a*f+b) = u(f). AISC learns a lightweight semantic correction
    from u and injects it according to the common-mode dominance:
        q = |mu| / (|mu| + s + eps)

    Input/output resolution and channels are unchanged.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        channels = int(channels)
        if channels < 1:
            raise ValueError(f"AISC channels must be positive, got {channels}.")
        self.channels = channels
        self.eps = float(eps)

        self.dw = nn.Conv2d(
            channels, channels, 3, 1, 1,
            groups=channels, bias=False
        )
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(
            channels, channels, 1, 1, 0,
            bias=False
        )

        # Exact identity at initialization, but the first backward can still
        # update dw because pw keeps its normal non-zero initialization.
        nn.init.zeros_(self.dw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"AISC expects BCHW input, got {tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"AISC expected {self.channels} channels, got {x.shape[1]}."
            )

        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        contrast = torch.sqrt(
            centered.square().mean(dim=1, keepdim=True) + self.eps
        )

        stable = centered / contrast

        dominance = mean.abs() / (mean.abs() + contrast + self.eps)
        dominance = F.avg_pool2d(dominance, 3, 1, 1)

        correction = self.pw(self.act(self.dw(stable)))
        return x + dominance * correction
