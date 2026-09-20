import torch
import torch.nn as nn


class AGRF(nn.Module):
    """
    Agreement-Guided Reliability Fusion (AGRF).

    AGRF replaces a two-input Concat node while preserving the original
    output channel count.

    Main idea:
        - compare two aligned scales through channel-invariant spatial evidence;
        - estimate their cross-scale agreement/disagreement;
        - use independent residual recalibration instead of competitive routing;
        - initialize exactly as the original Concat.

    No manually tuned alpha/beta fusion coefficient is used.

    Args:
        channels (list[int] | tuple[int, int]):
            Input channel counts of the two branches.
    """

    def __init__(self, channels):
        super().__init__()

        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError(
                f"AGRF expects exactly two input channel values, got {channels}."
            )

        c_a, c_b = int(channels[0]), int(channels[1])
        if c_a < 1 or c_b < 1:
            raise ValueError(
                f"AGRF input channels must be positive, got {channels}."
            )

        self.channels = (c_a, c_b)
        self.eps = 1e-6

        # Local spatial reliability estimator:
        # [relative_a, relative_b, disagreement, agreement] -> 2 residual logits.
        self.local_router = nn.Conv2d(
            4,
            2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # Global image-level reliability estimator.
        # Descriptor:
        # mean(|Ra-1|), mean(|Rb-1|), mean(D), mean(A), max(D), min(A)
        hidden = 8
        self.global_router = nn.Sequential(
            nn.Linear(6, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, 2, bias=True),
        )

        # Identity initialization:
        # residual = tanh(0) = 0 -> scale = 1.
        # Therefore AGRF starts exactly as standard Concat.
        nn.init.zeros_(self.local_router.weight)
        nn.init.zeros_(self.local_router.bias)
        nn.init.zeros_(self.global_router[-1].weight)
        nn.init.zeros_(self.global_router[-1].bias)

    @staticmethod
    def _check_inputs(x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            length = len(x) if isinstance(x, (list, tuple)) else "N/A"
            raise ValueError(
                "AGRF expects a list/tuple containing exactly two tensors, "
                f"got type={type(x).__name__}, len={length}."
            )

        x_a, x_b = x

        if x_a.ndim != 4 or x_b.ndim != 4:
            raise ValueError(
                "AGRF expects 4D BCHW tensors, "
                f"got shapes {tuple(x_a.shape)} and {tuple(x_b.shape)}."
            )

        if x_a.shape[0] != x_b.shape[0]:
            raise ValueError(
                "AGRF batch sizes must match, "
                f"got {x_a.shape[0]} and {x_b.shape[0]}."
            )

        if x_a.shape[-2:] != x_b.shape[-2:]:
            raise ValueError(
                "AGRF spatial sizes must already be aligned before fusion, "
                f"got {tuple(x_a.shape[-2:])} and {tuple(x_b.shape[-2:])}."
            )

        return x_a, x_b

    def _spatial_evidence(self, x):
        # Channel-invariant RMS spatial response.
        return torch.sqrt(
            x.pow(2).mean(dim=1, keepdim=True) + self.eps
        )

    def _relative_evidence(self, evidence):
        # Normalize by each image's own spatial mean.
        spatial_mean = evidence.mean(
            dim=(2, 3),
            keepdim=True,
        )
        return evidence / (spatial_mean + self.eps)

    def forward(self, x):
        x_a, x_b = self._check_inputs(x)

        # -------------------------------------------------------------
        # Channel-invariant spatial evidence
        # -------------------------------------------------------------
        evidence_a = self._spatial_evidence(x_a)
        evidence_b = self._spatial_evidence(x_b)

        relative_a = self._relative_evidence(evidence_a)
        relative_b = self._relative_evidence(evidence_b)

        # -------------------------------------------------------------
        # Cross-scale agreement / disagreement
        # -------------------------------------------------------------
        disagreement = (relative_a - relative_b).abs()

        # Fixed monotonic mapping:
        # D=0 -> A=1; larger disagreement -> smaller agreement.
        agreement = torch.exp(-disagreement)

        # -------------------------------------------------------------
        # Local reliability residual
        # -------------------------------------------------------------
        local_descriptor = torch.cat(
            (
                relative_a,
                relative_b,
                disagreement,
                agreement,
            ),
            dim=1,
        )
        local_logits = self.local_router(local_descriptor)

        # -------------------------------------------------------------
        # Global reliability residual
        # -------------------------------------------------------------
        deviation_a = (relative_a - 1.0).abs().mean(dim=(2, 3))
        deviation_b = (relative_b - 1.0).abs().mean(dim=(2, 3))
        mean_disagreement = disagreement.mean(dim=(2, 3))
        mean_agreement = agreement.mean(dim=(2, 3))
        max_disagreement = disagreement.amax(dim=(2, 3))
        min_agreement = agreement.amin(dim=(2, 3))

        global_descriptor = torch.cat(
            (
                deviation_a,
                deviation_b,
                mean_disagreement,
                mean_agreement,
                max_disagreement,
                min_agreement,
            ),
            dim=1,
        )

        global_logits = self.global_router(global_descriptor)
        global_logits = global_logits[:, :, None, None]

        # -------------------------------------------------------------
        # Independent residual recalibration
        # -------------------------------------------------------------
        # No softmax competition. Both branches may be enhanced, preserved,
        # or suppressed independently.
        residual_logits = local_logits + global_logits
        residual = torch.tanh(residual_logits)

        scale_a = 1.0 + residual[:, 0:1]
        scale_b = 1.0 + residual[:, 1:2]

        return torch.cat(
            (
                x_a * scale_a,
                x_b * scale_b,
            ),
            dim=1,
        )
