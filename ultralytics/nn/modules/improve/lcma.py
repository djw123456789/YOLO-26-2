import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCMA(nn.Module):
    """
    Lateral-Conditioned Matching Alignment (LCMA).

    Designed for YOLO26 top-down P4 -> P3 fusion.

    Inputs:
        [high_level_feature, lateral_feature]

    Steps:
        1) nearest-resize the high-level feature to the lateral resolution;
        2) use the lateral feature as a query;
        3) match the high-level feature inside a local 3x3 neighborhood;
        4) derive a parameter-free confidence from matching entropy;
        5) only perform a strong alignment when the local match is confident.

    No P2 head, no frequency transform, no deformable sampling,
    no learned offset, and no localization-loss modification.
    """

    def __init__(self, channels, eps: float = 1e-6):
        super().__init__()

        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError(
                f"LCMA expects channels=[high_channels, lateral_channels], got {channels}."
            )

        c_high, c_lateral = map(int, channels)
        if c_high != c_lateral:
            raise ValueError(
                "LCMA first-stage design requires equal high/lateral channels, "
                f"got {c_high} and {c_lateral}."
            )
        if c_high < 1:
            raise ValueError(f"LCMA channels must be positive, got {c_high}.")

        self.channels = c_high
        self.eps = float(eps)

        # Shared lightweight metric projection. Both branches use the same
        # projection, avoiding unrelated Q/K spaces.
        match_channels = max(16, c_high // 4)
        self.match_channels = match_channels
        self.metric = nn.Conv2d(
            c_high,
            match_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        nn.init.orthogonal_(self.metric.weight.view(match_channels, c_high))

        self.offsets = (
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1), ( 0, 0), ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        )
        self.register_buffer(
            "_log_candidates",
            torch.tensor(math.log(len(self.offsets)), dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Sample x at local offset (dy, dx) using replicate borders."""
        h, w = x.shape[-2:]
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        y0 = 1 + dy
        x0 = 1 + dx
        return x_pad[..., y0:y0 + h, x0:x0 + w]

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "LCMA expects two inputs: [high_level_feature, lateral_feature]."
            )

        high, lateral = x

        if high.ndim != 4 or lateral.ndim != 4:
            raise ValueError(
                f"LCMA expects BCHW tensors, got {high.shape} and {lateral.shape}."
            )

        if high.shape[1] != self.channels or lateral.shape[1] != self.channels:
            raise ValueError(
                f"LCMA expected {self.channels} channels for both inputs, "
                f"got {high.shape[1]} and {lateral.shape[1]}."
            )

        # Standard interpolation itself is not claimed as innovation.
        high_up = F.interpolate(
            high,
            size=lateral.shape[-2:],
            mode="nearest",
        )

        # Shared-metric local matching.
        q = self.metric(lateral)
        k = self.metric(high_up)

        q = F.normalize(q, p=2, dim=1, eps=self.eps)
        k = F.normalize(k, p=2, dim=1, eps=self.eps)

        scores = []
        for dy, dx in self.offsets:
            k_shift = self._shift(k, dy, dx)
            scores.append((q * k_shift).sum(dim=1, keepdim=True))

        scores = torch.cat(scores, dim=1)  # B, 9, H, W
        weights = torch.softmax(scores, dim=1)

        # Ambiguous matching -> high entropy -> confidence approaches 0.
        entropy = -(
            weights * torch.log(weights.clamp_min(self.eps))
        ).sum(dim=1, keepdim=True)

        confidence = 1.0 - entropy / self._log_candidates.to(
            device=entropy.device,
            dtype=entropy.dtype,
        )
        confidence = confidence.clamp(0.0, 1.0)

        # Sequential gather avoids F.unfold's large memory footprint at P3.
        matched = torch.zeros_like(high_up)
        for idx, (dy, dx) in enumerate(self.offsets):
            v_shift = self._shift(high_up, dy, dx)
            matched = matched + weights[:, idx:idx + 1] * v_shift

        # Parameter-free confidence-controlled residual alignment.
        return high_up + confidence * (matched - high_up)
