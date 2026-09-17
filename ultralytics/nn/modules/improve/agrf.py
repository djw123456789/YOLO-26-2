import torch
import torch.nn as nn
import torch.nn.functional as F


class AGRF(nn.Module):
    """
    Agreement-Guided Reliability Fusion (AGRF).

    AGRF is designed to replace a two-input Concat node in a PAN/FPN neck
    without changing the output channel count.

    Given two spatially aligned features Xa and Xb:
        1) Project both features into a lightweight shared embedding space.
        2) Estimate local cross-scale agreement using cosine similarity,
           response evidence, and embedding discrepancy.
        3) Estimate global branch reliability from pooled branch statistics.
        4) Combine local/global reliability logits and perform competitive
           softmax routing between the two branches.
        5) Use mean-one normalization so equal routing weights correspond to
           an identity-amplitude Concat:
               wa = wb = 1  at initialization.
        6) Concatenate the dynamically reweighted original features.

    This module does NOT use manually tuned alpha/beta fusion coefficients.

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

        # Lightweight shared reliability embedding.
        # The compression width is structurally derived from input channels;
        # it is not a manually tuned fusion coefficient.
        c_r = max(8, min(c_a, c_b) // 8)
        self.reliability_channels = c_r

        self.proj_a = nn.Sequential(
            nn.Conv2d(c_a, c_r, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_r),
        )

        self.proj_b = nn.Sequential(
            nn.Conv2d(c_b, c_r, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_r),
        )

        # -------------------------------------------------------------
        # Local agreement-aware reliability router
        # -------------------------------------------------------------
        # Four input maps:
        #   Ea   : branch-A absolute response evidence
        #   Eb   : branch-B absolute response evidence
        #   D    : cross-branch embedding discrepancy
        #   S    : cosine agreement
        #
        # Output:
        #   two spatial reliability logits, one for each branch.
        self.local_router = nn.Conv2d(
            4,
            2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # -------------------------------------------------------------
        # Global reliability router
        # -------------------------------------------------------------
        # Descriptor:
        #   GAP(|Za|), GAP(|Zb|), GAP(|Za - Zb|)
        self.global_router = nn.Sequential(
            nn.Linear(c_r * 3, c_r, bias=True),
            nn.SiLU(),
            nn.Linear(c_r, 2, bias=True),
        )

        # -------------------------------------------------------------
        # Concat-preserving initialization
        # -------------------------------------------------------------
        # Zero final routing logits -> softmax([0, 0]) = [0.5, 0.5].
        # Mean-one normalization in forward multiplies them by N=2,
        # therefore the initial modulation is [1, 1].
        #
        # This makes AGRF start from the amplitude behavior of the original
        # Concat instead of imposing a handcrafted preference.
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

    def forward(self, x):
        x_a, x_b = self._check_inputs(x)

        # Shared reliability embeddings.
        z_a = self.proj_a(x_a)
        z_b = self.proj_b(x_b)

        # -------------------------------------------------------------
        # Local cross-scale reliability evidence
        # -------------------------------------------------------------
        evidence_a = z_a.abs().mean(dim=1, keepdim=True)
        evidence_b = z_b.abs().mean(dim=1, keepdim=True)

        discrepancy = (z_a - z_b).abs().mean(dim=1, keepdim=True)

        z_a_norm = F.normalize(
            z_a,
            p=2,
            dim=1,
            eps=1e-6,
        )
        z_b_norm = F.normalize(
            z_b,
            p=2,
            dim=1,
            eps=1e-6,
        )

        agreement = (z_a_norm * z_b_norm).sum(
            dim=1,
            keepdim=True,
        )

        local_descriptor = torch.cat(
            (
                evidence_a,
                evidence_b,
                discrepancy,
                agreement,
            ),
            dim=1,
        )

        local_logits = self.local_router(local_descriptor)

        # -------------------------------------------------------------
        # Global branch reliability
        # -------------------------------------------------------------
        global_a = F.adaptive_avg_pool2d(
            z_a.abs(),
            output_size=1,
        ).flatten(1)

        global_b = F.adaptive_avg_pool2d(
            z_b.abs(),
            output_size=1,
        ).flatten(1)

        global_diff = F.adaptive_avg_pool2d(
            (z_a - z_b).abs(),
            output_size=1,
        ).flatten(1)

        global_descriptor = torch.cat(
            (
                global_a,
                global_b,
                global_diff,
            ),
            dim=1,
        )

        global_logits = self.global_router(global_descriptor)
        global_logits = global_logits[:, :, None, None]

        # -------------------------------------------------------------
        # Competitive reliability routing
        # -------------------------------------------------------------
        route_logits = local_logits + global_logits

        route_weight = torch.softmax(
            route_logits,
            dim=1,
        )

        # Mean-one normalization:
        # for two branches, uniform softmax 0.5 -> modulation weight 1.0.
        route_weight = route_weight * route_weight.shape[1]

        weight_a = route_weight[:, 0:1]
        weight_b = route_weight[:, 1:2]

        # Reweight original (not compressed) features, preserving information
        # capacity and keeping output channels identical to standard Concat.
        out = torch.cat(
            (
                x_a * weight_a,
                x_b * weight_b,
            ),
            dim=1,
        )

        return out
