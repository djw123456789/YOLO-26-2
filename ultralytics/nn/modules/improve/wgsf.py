import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conv import Conv


class AdaptiveHaarCut(nn.Module):
    """
    Adaptive Haar Wavelet Downsampling.

    The input feature is decomposed into four orthogonal Haar sub-bands:
        LL: low-frequency / coarse structure
        LH: horizontal detail
        HL: vertical detail
        HH: diagonal detail

    Different from using four global learnable scalar weights, the importance
    of the four bands is predicted dynamically from the current input feature.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()

        self.c1 = c1
        self.c2 = c2

        # Frequency descriptor:
        # [LL, LH, HL, HH] -> global band statistics -> 4 adaptive weights.
        #
        # Hidden width is structurally derived from the input channel count,
        # not used as a manually tuned fusion coefficient.
        hidden = max(c1, 4)

        self.band_router = nn.Sequential(
            nn.Linear(c1 * 4, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, 4, bias=True),
        )

        # Start from equal treatment of all four frequency bands.
        # The actual weights become input-adaptive during training.
        nn.init.zeros_(self.band_router[-1].weight)
        nn.init.zeros_(self.band_router[-1].bias)

        # Fuse four Haar bands.
        self.fuse = Conv(
            c1 * 4,
            c2,
            k=1,
            s=1,
        )

    @staticmethod
    def _pad_to_even(x: torch.Tensor) -> torch.Tensor:
        """
        Pad feature maps whose H/W are odd.

        This keeps the Haar branch spatial size consistent with stride-2
        convolution: output size is effectively ceil(H / 2), ceil(W / 2).
        """
        h, w = x.shape[-2:]

        pad_h = h % 2
        pad_w = w % 2

        if pad_h or pad_w:
            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="replicate",
            )

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_to_even(x)

        # ---------------------------------------------------------
        # Haar decomposition
        # ---------------------------------------------------------
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # Orthonormal 2D Haar transform.
        LL = (x00 + x01 + x10 + x11) * 0.5
        LH = (x00 - x01 + x10 - x11) * 0.5
        HL = (x00 + x01 - x10 - x11) * 0.5
        HH = (x00 - x01 - x10 + x11) * 0.5

        bands = (LL, LH, HL, HH)

        # ---------------------------------------------------------
        # Dynamic frequency-band estimation
        # ---------------------------------------------------------
        #
        # Use absolute spectral response because the sign of Haar
        # coefficients represents direction while magnitude represents
        # response strength.
        descriptors = [
            F.adaptive_avg_pool2d(band.abs(), output_size=1).flatten(1)
            for band in bands
        ]

        descriptor = torch.cat(descriptors, dim=1)

        # [B, 4]
        band_logits = self.band_router(descriptor)

        # Competitive selection between LL/LH/HL/HH.
        band_weight = torch.softmax(
            band_logits,
            dim=1,
        )

        # Mean-one normalization.
        #
        # softmax gives sum(weight)=1. Since there are N frequency bands,
        # multiplying by N makes uniform initialization correspond to
        # weight=1 for every band.
        #
        # N is mathematically determined by the number of Haar bands,
        # not an empirically selected alpha/beta coefficient.
        band_weight = band_weight * band_weight.shape[1]

        # ---------------------------------------------------------
        # Adaptive band modulation
        # ---------------------------------------------------------
        weighted_bands = []

        for i, band in enumerate(bands):
            w = band_weight[:, i].view(-1, 1, 1, 1)
            weighted_bands.append(band * w)

        x_wave = torch.cat(weighted_bands, dim=1)

        return self.fuse(x_wave)


class AdaptiveWaveletDownsample(nn.Module):
    """
    Adaptive Spatial-Wavelet Downsampling.

    Two complementary downsampling paths are constructed:

        Spatial path:
            learnable 3x3 stride-2 convolution

        Wavelet path:
            adaptive Haar decomposition

    A data-dependent router determines how much information should come
    from each branch for every input image.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()

        self.c1 = c1
        self.c2 = c2

        # ---------------------------------------------------------
        # Spatial branch
        # ---------------------------------------------------------
        #
        # Keep a conventional learnable downsampling path so that
        # clean/simple scenes do not have to rely entirely on wavelets.
        self.spatial_branch = Conv(
            c1,
            c2,
            k=3,
            s=2,
        )

        # ---------------------------------------------------------
        # Wavelet branch
        # ---------------------------------------------------------
        self.wavelet_branch = AdaptiveHaarCut(
            c1,
            c2,
        )

        # ---------------------------------------------------------
        # Spatial / Wavelet router
        # ---------------------------------------------------------
        hidden = max(c2 // 4, 4)

        self.branch_router = nn.Sequential(
            nn.Linear(c2 * 2, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, 2, bias=True),
        )

        # Equal initial logits:
        # no manually imposed preference for spatial/wavelet branches.
        nn.init.zeros_(self.branch_router[-1].weight)
        nn.init.zeros_(self.branch_router[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conventional spatial downsampling.
        x_spatial = self.spatial_branch(x)

        # Frequency-preserving adaptive wavelet downsampling.
        x_wavelet = self.wavelet_branch(x)

        # ---------------------------------------------------------
        # Dynamic degradation-aware routing
        # ---------------------------------------------------------
        spatial_desc = F.adaptive_avg_pool2d(
            x_spatial,
            output_size=1,
        ).flatten(1)

        wavelet_desc = F.adaptive_avg_pool2d(
            x_wavelet,
            output_size=1,
        ).flatten(1)

        descriptor = torch.cat(
            (spatial_desc, wavelet_desc),
            dim=1,
        )

        # [B, 2]
        route_logits = self.branch_router(descriptor)

        route_weight = torch.softmax(
            route_logits,
            dim=1,
        )

        spatial_weight = route_weight[:, 0].view(-1, 1, 1, 1)
        wavelet_weight = route_weight[:, 1].view(-1, 1, 1, 1)

        # Input-adaptive convex fusion.
        out = (
            spatial_weight * x_spatial
            + wavelet_weight * x_wavelet
        )

        return out


class WGFS(nn.Module):
    """
    Adaptive Wavelet-Guided Feature Sampling V2.

    Designed as a replacement for the first TWO stride-2 convolutions
    in YOLO26.

    Spatial resolution:
        H x W
          -> Stage 1 -> H/2 x W/2
          -> Stage 2 -> H/4 x W/4

    Therefore WGFS directly outputs a P2/4 feature map.

    Main ideas:
        1. Haar decomposition avoids directly discarding frequency content.
        2. LL/LH/HL/HH importance is input-adaptive.
        3. Spatial and wavelet paths are dynamically routed.
        4. No MaxPool branch is used to reduce amplification of adverse
           weather/background peak responses.
        5. No manually specified fusion coefficients are used.

    Args:
        c1 (int): Input channels.
        c2 (int): Final output channels.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
    ):
        super().__init__()

        # This naturally reproduces the channel progression of the
        # original two-stage YOLO stem:
        #
        # c1 -> c2/2 -> c2
        #
        # Example for YOLO26n after width scaling:
        # 3 -> 16 -> 32.
        c_mid = c2 // 2

        if c_mid < 1:
            raise ValueError(
                f"WGFS requires c2 >= 2, but got c2={c2}."
            )

        self.stage1 = AdaptiveWaveletDownsample(
            c1,
            c_mid,
        )

        self.stage2 = AdaptiveWaveletDownsample(
            c_mid,
            c2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)

        return x