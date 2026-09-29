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

    Main procedure:
        1. Resize high-level feature to the lateral resolution.
        2. Use lateral feature as local matching query.
        3. Search a 3x3 neighborhood in the upsampled high-level feature.
        4. Perform shared-metric cosine matching.
        5. Calibrate cosine logits by sqrt(d).
        6. Estimate matching confidence from normalized entropy.
        7. Apply confidence-controlled residual alignment.

    The sqrt(d) calibration compensates for the naturally small
    variance of cosine similarities in a d-dimensional normalized
    embedding space. It introduces no learnable temperature and
    no manually tuned scaling hyperparameter.

    Design boundaries:
        - no P2 detection head
        - no FFT / wavelet / frequency transform
        - no deformable convolution
        - no learned spatial offset
        - no grid_sample
        - no localization-loss modification
        - no manually tuned threshold / alpha / beta
        - only P4 -> P3 top-down fusion is modified

    Args:
        channels:
            [high_level_channels, lateral_channels]

            The current lightweight design requires both branches
            to have equal channel dimensions after YOLO scaling.

        eps:
            Numerical stability constant.
    """

    def __init__(self, channels, eps: float = 1e-6):
        super().__init__()

        # ---------------------------------------------------------
        # 1. Validate input-channel specification
        # ---------------------------------------------------------
        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError(
                "LCMA expects channels=[high_channels, lateral_channels], "
                f"but got {channels}."
            )

        c_high, c_lateral = map(int, channels)

        if c_high != c_lateral:
            raise ValueError(
                "LCMA requires equal high/lateral channels in the "
                f"current design, but got {c_high} and {c_lateral}."
            )

        if c_high < 1:
            raise ValueError(
                f"LCMA channels must be positive, got {c_high}."
            )

        self.channels = c_high
        self.eps = float(eps)

        # ---------------------------------------------------------
        # 2. Lightweight shared matching space
        #
        # Both high-level and lateral features use the SAME
        # projection. This avoids learning two unrelated Q/K spaces
        # and keeps parameter cost very small.
        #
        # YOLO26n P4->P3 case:
        #   channels = 128
        #   match_channels = 32
        #   parameters = 128 * 32 = 4096
        # ---------------------------------------------------------
        self.match_channels = max(
            16,
            self.channels // 4,
        )

        self.metric = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.match_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Stable initial shared metric space.
        nn.init.orthogonal_(
            self.metric.weight.view(
                self.match_channels,
                self.channels,
            )
        )

        # ---------------------------------------------------------
        # 3. Fixed 3x3 local matching candidates
        #
        # No learned offsets are introduced.
        # ---------------------------------------------------------
        self.offsets = (
            (-1, -1),
            (-1,  0),
            (-1,  1),

            ( 0, -1),
            ( 0,  0),
            ( 0,  1),

            ( 1, -1),
            ( 1,  0),
            ( 1,  1),
        )

        # Used for normalized entropy:
        #
        #   H(a) / log(9)
        #
        # so confidence naturally lies in [0, 1].
        self.register_buffer(
            "_log_candidates",
            torch.tensor(
                math.log(len(self.offsets)),
                dtype=torch.float32,
            ),
            persistent=False,
        )

        # ---------------------------------------------------------
        # 4. Fixed dimensional calibration factor
        #
        # q and k are L2-normalized.
        #
        # For approximately isotropic normalized d-dimensional
        # vectors, cosine-similarity variance scales approximately
        # as 1/d.
        #
        # Multiplying by sqrt(d) restores an O(1) logit scale.
        #
        # IMPORTANT:
        # This is mathematically determined from embedding dimension,
        # not a manually tuned temperature.
        # ---------------------------------------------------------
        self.logit_scale = math.sqrt(
            self.match_channels
        )

    @staticmethod
    def _shift(
        x: torch.Tensor,
        dy: int,
        dx: int,
    ) -> torch.Tensor:
        """
        Sample feature x at a fixed local spatial offset.

        Replicate padding is used to avoid artificial zero-valued
        borders.

        Args:
            x:
                Tensor of shape [B, C, H, W].

            dy:
                Vertical offset in {-1, 0, 1}.

            dx:
                Horizontal offset in {-1, 0, 1}.

        Returns:
            Shifted feature with unchanged shape.
        """

        h, w = x.shape[-2:]

        x_pad = F.pad(
            x,
            pad=(1, 1, 1, 1),
            mode="replicate",
        )

        y0 = 1 + dy
        x0 = 1 + dx

        return x_pad[
            ...,
            y0:y0 + h,
            x0:x0 + w,
        ]

    def forward(self, x):
        """
        Args:
            x:
                [
                    high_level_feature,
                    lateral_feature
                ]

                high:
                    lower-resolution semantic feature,
                    e.g. top-down P4.

                lateral:
                    higher-resolution backbone feature,
                    e.g. backbone P3.

        Returns:
            Spatially aligned high-level feature with the same
            resolution and channels as lateral.
        """

        # ---------------------------------------------------------
        # 1. Input checking
        # ---------------------------------------------------------
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "LCMA expects exactly two inputs: "
                "[high_level_feature, lateral_feature]."
            )

        high, lateral = x

        if high.ndim != 4 or lateral.ndim != 4:
            raise ValueError(
                "LCMA expects BCHW feature tensors, "
                f"but got {tuple(high.shape)} and "
                f"{tuple(lateral.shape)}."
            )

        if high.shape[0] != lateral.shape[0]:
            raise ValueError(
                "LCMA batch-size mismatch: "
                f"{high.shape[0]} vs {lateral.shape[0]}."
            )

        if (
            high.shape[1] != self.channels
            or lateral.shape[1] != self.channels
        ):
            raise ValueError(
                f"LCMA expects {self.channels} channels "
                "for both inputs, but got "
                f"{high.shape[1]} and {lateral.shape[1]}."
            )

        # =========================================================
        # Step 1:
        # Standard top-down resizing
        # =========================================================
        #
        # Interpolation itself is NOT treated as the innovation.
        # LCMA focuses on correcting local semantic correspondence
        # after resizing and before fusion.
        #
        high_up = F.interpolate(
            high,
            size=lateral.shape[-2:],
            mode="nearest",
        )

        # =========================================================
        # Step 2:
        # Shared metric projection
        # =========================================================
        #
        # lateral:
        #     query / spatial reference
        #
        # high_up:
        #     semantic candidates to be locally aligned
        #
        q = self.metric(lateral)
        k = self.metric(high_up)

        # Cosine matching space.
        q = F.normalize(
            q,
            p=2,
            dim=1,
            eps=self.eps,
        )

        k = F.normalize(
            k,
            p=2,
            dim=1,
            eps=self.eps,
        )

        # =========================================================
        # Step 3:
        # Local 3x3 correspondence matching
        # =========================================================
        scores = []

        for dy, dx in self.offsets:

            k_shift = self._shift(
                k,
                dy,
                dx,
            )

            # Cosine similarity because q and k are normalized.
            score = (
                q * k_shift
            ).sum(
                dim=1,
                keepdim=True,
            )

            scores.append(score)

        # Shape:
        #   [B, 9, H, W]
        scores = torch.cat(
            scores,
            dim=1,
        )

        # =========================================================
        # Step 4:
        # Dimension-aware logit calibration
        # =========================================================
        #
        # Original LCMA:
        #
        #   softmax(cosine_similarity)
        #
        # Because cosine similarity is naturally concentrated near
        # zero in a moderately high-dimensional normalized space,
        # the nine logits can become too similar.
        #
        # That makes softmax overly flat:
        #
        #   weights ~= uniform
        #
        # which in turn makes entropy high and the later confidence
        # gate very small.
        #
        # We therefore apply:
        #
        #   s_hat = sqrt(d) * s
        #
        # where:
        #
        #   d = match_channels
        #
        # No trainable or manually tuned temperature is used.
        #
        scores = (
            scores
            * self.logit_scale
        )

        # Local correspondence probability.
        weights = torch.softmax(
            scores,
            dim=1,
        )

        # =========================================================
        # Step 5:
        # Entropy-derived correspondence confidence
        # =========================================================
        #
        # If all nine candidates are almost equally likely:
        #
        #   entropy ~= log(9)
        #   confidence ~= 0
        #
        # LCMA therefore avoids an unreliable alignment.
        #
        # If matching is concentrated around one/few candidates:
        #
        #   entropy decreases
        #   confidence increases
        #
        entropy = -(
            weights
            * torch.log(
                weights.clamp_min(self.eps)
            )
        ).sum(
            dim=1,
            keepdim=True,
        )

        log_candidates = (
            self._log_candidates
            .to(
                device=entropy.device,
                dtype=entropy.dtype,
            )
        )

        confidence = (
            1.0
            - entropy / log_candidates
        )

        confidence = confidence.clamp(
            min=0.0,
            max=1.0,
        )

        # =========================================================
        # Step 6:
        # Gather locally matched semantic feature
        # =========================================================
        #
        # Do NOT use F.unfold here.
        #
        # On 80x80 P3 feature maps, unfold creates a much larger
        # temporary tensor and increases GPU memory consumption.
        #
        # Sequential shifts are slower than a custom CUDA kernel,
        # but much more memory friendly and easy to reproduce.
        #
        matched = torch.zeros_like(
            high_up
        )

        for idx, (dy, dx) in enumerate(
            self.offsets
        ):
            value_shift = self._shift(
                high_up,
                dy,
                dx,
            )

            matched = (
                matched
                + weights[:, idx:idx + 1]
                * value_shift
            )

        # =========================================================
        # Step 7:
        # Confidence-controlled residual alignment
        # =========================================================
        #
        # equivalent to:
        #
        #   aligned =
        #       (1 - confidence) * high_up
        #       + confidence * matched
        #
        # Written in residual form:
        #
        #   aligned =
        #       high_up
        #       + confidence * (matched - high_up)
        #
        # This prevents uncertain local matches from aggressively
        # destroying the original semantic feature.
        #
        aligned = (
            high_up
            + confidence
            * (
                matched
                - high_up
            )
        )

        return aligned