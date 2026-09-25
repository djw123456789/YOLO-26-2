import torch
import torch.nn as nn
import torch.nn.functional as F


class DRCC(nn.Module):
    """
    Dual-Ring Context Consensus (DRCC).

    A location is enhanced only when:
      1) the immediate surround and farther surround agree with each other,
         i.e. they form a reliable local-background context; and
      2) the center feature is distinctly different from both surrounds.

    near: mean of the 3x3 ring excluding center (8 samples)
    far:  mean of the outer 5x5 ring excluding inner 3x3 (16 samples)

    Input/output shape is unchanged.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        channels = int(channels)
        if channels < 1:
            raise ValueError(f"DRCC channels must be positive, got {channels}.")

        self.channels = channels
        self.eps = float(eps)

        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            groups=channels, bias=False
        )
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(
            channels, channels, kernel_size=1, stride=1, padding=0,
            bias=False
        )

        # Exact identity at initialization while preserving first-step gradient
        # flow into the depthwise kernel through the non-zero pointwise kernel.
        nn.init.zeros_(self.dw.weight)

    @staticmethod
    def _ring_contexts(x: torch.Tensor):
        # Replicate padding avoids artificial zero-valued border contrast.
        x3 = F.pad(x, (1, 1, 1, 1), mode="replicate")
        x5 = F.pad(x, (2, 2, 2, 2), mode="replicate")

        sum3 = F.avg_pool2d(x3, kernel_size=3, stride=1) * 9.0
        sum5 = F.avg_pool2d(x5, kernel_size=5, stride=1) * 25.0

        near = (sum3 - x) / 8.0
        far = (sum5 - sum3) / 16.0
        return near, far

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.square().mean(dim=1, keepdim=True) + self.eps)

    def _channel_direction(self, x: torch.Tensor) -> torch.Tensor:
        # Remove common channel bias, then normalize the semantic direction.
        centered = x - x.mean(dim=1, keepdim=True)
        return centered / self._rms(centered)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DRCC expects BCHW input, got shape={tuple(x.shape)}.")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"DRCC channel mismatch: expected {self.channels}, got {x.shape[1]}."
            )

        # -------------------------------------------------------------
        # 1. Two independent surrounding context prototypes
        # -------------------------------------------------------------
        near, far = self._ring_contexts(x)

        # -------------------------------------------------------------
        # 2. Background-context consensus
        #
        # This compares near vs. far surrounds themselves, rather than
        # comparing two center-containing residuals. Therefore a random or
        # cluttered center does not automatically produce high consensus.
        # -------------------------------------------------------------
        u_near_ctx = self._channel_direction(near)
        u_far_ctx = self._channel_direction(far)

        context_consensus = (u_near_ctx * u_far_ctx).mean(
            dim=1, keepdim=True
        ).clamp(0.0, 1.0)

        # -------------------------------------------------------------
        # 3. Center distinctiveness against BOTH surrounds
        # -------------------------------------------------------------
        r_near = x - near
        r_far = x - far

        e_near = self._rms(r_near)
        e_far = self._rms(r_far)

        x_energy = self._rms(x)
        near_energy = self._rms(near)
        far_energy = self._rms(far)

        d_near = e_near / (x_energy + near_energy + self.eps)
        d_far = e_far / (x_energy + far_energy + self.eps)

        # Geometric mean: both contexts must support the deviation.
        distinctiveness = torch.sqrt((d_near * d_far).clamp(min=0.0))

        gate = context_consensus * distinctiveness

        # -------------------------------------------------------------
        # 4. Semantic deviation from the consensus background prototype
        # -------------------------------------------------------------
        background = 0.5 * (near + far)
        direction = x - background
        direction = direction / self._rms(direction)

        # -------------------------------------------------------------
        # 5. Lightweight semantic correction
        # -------------------------------------------------------------
        correction = self.pw(self.act(self.dw(direction)))
        return x + gate * correction
