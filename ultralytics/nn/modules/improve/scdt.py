import torch
import torch.nn as nn
import torch.nn.functional as F


class SCDT(nn.Module):
    """
    Semantic-Conditioned Detail Transport (SCDT).

    SCDT uses a high-resolution shallow feature as a detail reservoir and lets
    the semantically stronger P3 feature retrieve useful sub-pixel details.

    The module expects exactly two inputs:
        x_detail: [B, C_d, 2H, 2W]  -- high-resolution P2 feature
        x_sem:    [B, C_s,  H,  W]  -- semantic P3 feature

    Main steps:
        1. Losslessly rearrange each 2x2 P2 neighborhood into four aligned
           sub-pixel feature vectors at the P3 grid.
        2. Remove the local common component and retain four sub-pixel detail
           residuals.
        3. Build a semantic query from P3 and shared keys from the four detail
           residuals.
        4. Perform four-way local routing to retrieve the detail residual that
           is most compatible with the P3 semantics at each spatial location.
        5. Inject the retrieved detail through a zero-initialized projection:

               Y = x_sem + Proj(D_retrieved)

           so the module is exactly an identity mapping at initialization.

    This module does not add a P2 detection head and does not change the P3
    output resolution or channel count.

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

        # Lightweight joint query-key space.
        # Width is structurally derived from the input channels rather than
        # introduced as a manually tuned fusion coefficient.
        c_attn = max(8, min(c_detail, c_sem) // 8)
        self.attn_channels = c_attn
        self.scale = c_attn ** -0.5

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

        # One shared key projection is used for all four sub-pixel positions.
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

        # Zero-initialized residual projection makes SCDT exactly reduce to
        # the original P3 branch at the start of training.
        self.out_proj = nn.Conv2d(
            c_detail,
            c_sem,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        nn.init.zeros_(self.out_proj.weight)

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
        Losslessly map [B, C, 2H, 2W] to [B, 4, C, H, W].

        The four groups correspond to the four spatial positions inside each
        2x2 neighborhood. No averaging or strided sampling is performed.
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
        # 1. Lossless 2x2 sub-pixel decomposition
        # -------------------------------------------------------------
        groups = self._split_subpixels(
            x_detail,
            h,
            w,
        )  # [B, 4, C_d, H, W]

        # -------------------------------------------------------------
        # 2. Isolate sub-pixel detail residuals
        # -------------------------------------------------------------
        # The local common component is already largely represented by the
        # lower-resolution semantic path. We transport only within-cell
        # deviations to avoid redundantly injecting shallow low-frequency
        # responses.
        local_common = groups.mean(
            dim=1,
            keepdim=True,
        )
        detail = groups - local_common

        # -------------------------------------------------------------
        # 3. Semantic query and shared detail keys
        # -------------------------------------------------------------
        query = self.query(x_sem)  # [B, C_a, H, W]

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
        # 4. Semantic-conditioned four-way sub-pixel retrieval
        # -------------------------------------------------------------
        scores = (
            keys * query.unsqueeze(1)
        ).sum(
            dim=2
        ) * self.scale  # [B, 4, H, W]

        route = torch.softmax(
            scores,
            dim=1,
        )

        retrieved_detail = (
            detail * route.unsqueeze(2)
        ).sum(
            dim=1
        )  # [B, C_d, H, W]

        # -------------------------------------------------------------
        # 5. Identity-safe residual detail injection
        # -------------------------------------------------------------
        residual = self.out_proj(retrieved_detail)

        return x_sem + residual
