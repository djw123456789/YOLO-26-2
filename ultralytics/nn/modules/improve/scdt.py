import torch
import torch.nn as nn


class SCDT(nn.Module):
    """
    Semantic-Conditioned Detail Transport (SCDT).

    The high-resolution P2 feature is treated as a detail reservoir.
    P3 semantics dynamically retrieve sub-pixel detail residuals.

    Important initialization:
        - the four sub-pixel residuals sum to zero;
        - the semantic query projection is initialized to zero, so all four
          routing logits are exactly equal at initialization;
        - therefore routing is exactly uniform and retrieved_detail == 0;
        - out_proj remains normally initialized (non-zero), so gradients can
          flow from the detection loss into the routing branch immediately.

    Thus:
        output == x_sem exactly at initialization,
    while avoiding the gradient starvation caused by a zero-initialized
    residual output projection.

    Args:
        channels (list[int] | tuple[int, int]):
            [detail_channels, semantic_channels]
    """

    def __init__(self, channels):
        super().__init__()

        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError(
                f"SCDT expects [detail_channels, semantic_channels], got {channels}."
            )

        c_detail, c_sem = int(channels[0]), int(channels[1])

        if c_detail < 1 or c_sem < 1:
            raise ValueError(
                f"SCDT channels must be positive, got {channels}."
            )

        self.detail_channels = c_detail
        self.semantic_channels = c_sem

        # Lightweight shared query-key space.
        c_attn = max(8, min(c_detail, c_sem) // 8)
        self.attn_channels = c_attn
        self.scale = c_attn ** -0.5

        # Semantic query.
        # Zeroing ONLY the convolution weight makes q == 0 initially.
        # Hence all four dot-product logits are 0 -> exact uniform routing.
        #
        # Unlike zero-initializing out_proj, this still permits a non-zero
        # gradient to query.weight on the first backward pass because keys and
        # out_proj are both non-zero.
        self.query = nn.Sequential(
            nn.Conv2d(
                c_sem,
                c_attn,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(c_attn),
        )

        # Shared key projection for all four sub-pixel detail residuals.
        self.key = nn.Sequential(
            nn.Conv2d(
                c_detail,
                c_attn,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(c_attn),
        )

        # IMPORTANT:
        # Keep out_proj with normal PyTorch initialization.
        # Do NOT zero-initialize this layer.
        self.out_proj = nn.Conv2d(
            c_detail,
            c_sem,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Identity-safe but gradient-active initialization:
        # q = 0 -> route_i = 1/4.
        # Since sum_i detail_i = 0, retrieved_detail = 0 exactly.
        nn.init.zeros_(self.query[0].weight)

    @staticmethod
    def _check_inputs(x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            length = len(x) if isinstance(x, (list, tuple)) else "N/A"
            raise ValueError(
                "SCDT expects a list/tuple containing exactly two tensors, "
                f"got type={type(x).__name__}, len={length}."
            )

        x_detail, x_sem = x

        if x_detail.ndim != 4 or x_sem.ndim != 4:
            raise ValueError(
                "SCDT expects 4D BCHW tensors, "
                f"got {tuple(x_detail.shape)} and {tuple(x_sem.shape)}."
            )

        if x_detail.shape[0] != x_sem.shape[0]:
            raise ValueError(
                "SCDT batch sizes must match, "
                f"got {x_detail.shape[0]} and {x_sem.shape[0]}."
            )

        h_d, w_d = x_detail.shape[-2:]
        h_s, w_s = x_sem.shape[-2:]

        if h_d != 2 * h_s or w_d != 2 * w_s:
            raise ValueError(
                "SCDT expects the detail feature to have exactly 2x the "
                "semantic feature resolution, "
                f"got detail={(h_d, w_d)}, semantic={(h_s, w_s)}."
            )

        return x_detail, x_sem

    @staticmethod
    def _split_subpixels(x_detail, h, w):
        """
        Losslessly map [B, C, 2H, 2W] -> [B, 4, C, H, W].
        """
        b, c, _, _ = x_detail.shape

        return (
            x_detail.reshape(b, c, h, 2, w, 2)
            .permute(0, 3, 5, 1, 2, 4)
            .contiguous()
            .reshape(b, 4, c, h, w)
        )

    def forward(self, x):
        x_detail, x_sem = self._check_inputs(x)

        b, _, h, w = x_sem.shape

        # -------------------------------------------------------------
        # 1. Lossless sub-pixel decomposition
        # -------------------------------------------------------------
        groups = self._split_subpixels(
            x_detail,
            h,
            w,
        )

        # -------------------------------------------------------------
        # 2. Zero-sum within-cell detail residuals
        # -------------------------------------------------------------
        local_common = groups.mean(
            dim=1,
            keepdim=True,
        )
        detail = groups - local_common

        # -------------------------------------------------------------
        # 3. Semantic query and detail keys
        # -------------------------------------------------------------
        query = self.query(x_sem)

        keys = self.key(
            detail.reshape(
                b * 4,
                self.detail_channels,
                h,
                w,
            )
        )

        keys = keys.reshape(
            b,
            4,
            self.attn_channels,
            h,
            w,
        )

        # -------------------------------------------------------------
        # 4. Semantic-conditioned sub-pixel routing
        # -------------------------------------------------------------
        scores = (
            keys * query.unsqueeze(1)
        ).sum(
            dim=2
        ) * self.scale

        route = torch.softmax(
            scores,
            dim=1,
        )

        retrieved_detail = (
            detail * route.unsqueeze(2)
        ).sum(
            dim=1
        )

        # -------------------------------------------------------------
        # 5. Residual injection
        # -------------------------------------------------------------
        residual = self.out_proj(
            retrieved_detail
        )

        return x_sem + residual
