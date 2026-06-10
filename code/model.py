"""CVAF-SwinB: Cross-View Asymmetric Fusion Swin Transformer.

Binary breast cancer classification from four mammographic views
(L-CC, L-MLO, R-CC, R-MLO).

Stage 1  SwinBackbone      shared Swin-B encoder, all four views in one pass
Stage 2  LateralFusion     asymmetric CC<->MLO attention within each breast
Stage 3  BilateralFusion   contralateral L<->R attention between breasts
Stage 4  GlobalAggregator   gated per-view weighting + classification head
"""

from typing import List

import torch
import torch.nn as nn
from timm import create_model


class FFN(nn.Module):
    """Two-layer feed-forward block with GELU activation."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwinBackbone(nn.Module):
    """Swin-B encoder shared across all four views.

    Input  : (B, 3, H, W)
    Output : (B, N, C) with N = 49 spatial tokens and C = 1024.
    """

    def __init__(self, pretrained: bool = True, img_size: int = 224):
        super().__init__()
        self.swin = create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=img_size,
        )
        self.feature_dim = self.swin.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.swin.forward_features(x)
        if out.dim() == 4:
            B, H, W, C = out.shape
            out = out.view(B, H * W, C)
        return out


class LateralFusion(nn.Module):
    """Asymmetric cross-attention between CC and MLO views of one breast.

    Step 1: CC queries MLO, so CC absorbs oblique-view context.
    Step 2: MLO queries the updated CC, so MLO absorbs structural context.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cc_to_mlo = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cc1 = nn.LayerNorm(dim)
        self.norm_cc2 = nn.LayerNorm(dim)
        self.ffn_cc = FFN(dim, dropout=dropout)

        self.mlo_to_cc = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlo1 = nn.LayerNorm(dim)
        self.norm_mlo2 = nn.LayerNorm(dim)
        self.ffn_mlo = FFN(dim, dropout=dropout)

    def forward(self, cc: torch.Tensor, mlo: torch.Tensor):
        cc_att, _ = self.cc_to_mlo(query=cc, key=mlo, value=mlo)
        cc = self.norm_cc1(cc + cc_att)
        cc = self.norm_cc2(cc + self.ffn_cc(cc))

        mlo_att, _ = self.mlo_to_cc(query=mlo, key=cc, value=cc)
        mlo = self.norm_mlo1(mlo + mlo_att)
        mlo = self.norm_mlo2(mlo + self.ffn_mlo(mlo))
        return cc, mlo


class BilateralFusion(nn.Module):
    """Contralateral cross-attention between left and right breast views.

    Operates on pooled global vectors then broadcasts the attended context
    back onto the full spatial token sequence as a residual.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_l = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_r = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_l1 = nn.LayerNorm(dim)
        self.norm_l2 = nn.LayerNorm(dim)
        self.norm_r1 = nn.LayerNorm(dim)
        self.norm_r2 = nn.LayerNorm(dim)
        self.ffn_l = FFN(dim, dropout=dropout)
        self.ffn_r = FFN(dim, dropout=dropout)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        lg = left.mean(dim=1, keepdim=True)
        rg = right.mean(dim=1, keepdim=True)

        lc, _ = self.cross_l(query=lg, key=rg, value=rg)
        lg = self.norm_l1(lg + lc)
        lg = self.norm_l2(lg + self.ffn_l(lg))

        rc, _ = self.cross_r(query=rg, key=lg, value=lg)
        rg = self.norm_r1(rg + rc)
        rg = self.norm_r2(rg + self.ffn_r(rg))

        return left + lg, right + rg


class GlobalAggregator(nn.Module):
    """Gated per-view weighting followed by a classification head."""

    def __init__(self, dim: int, num_views: int = 4,
                 num_classes: int = 1, dropout: float = 0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_views, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_views),
            nn.Softmax(dim=-1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        pooled = [f.mean(dim=1) for f in feats]
        concat = torch.cat(pooled, dim=-1)
        gates = self.gate(concat)
        stacked = torch.stack(pooled, dim=1)
        fused = (gates.unsqueeze(-1) * stacked).sum(dim=1)
        return self.head(fused)


class CVAFSwinB(nn.Module):
    """Complete CVAF-SwinB model combining all four stages."""

    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 224,
        pretrained: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=pretrained, img_size=img_size)
        C = self.backbone.feature_dim

        self.lateral_left = LateralFusion(C, num_heads, dropout)
        self.lateral_right = LateralFusion(C, num_heads, dropout)

        self.bilateral_cc = BilateralFusion(C, num_heads, dropout)
        self.bilateral_mlo = BilateralFusion(C, num_heads, dropout)

        self.aggregator = GlobalAggregator(
            C, num_views=4, num_classes=num_classes, dropout=head_dropout)

    def forward(
        self,
        lcc: torch.Tensor,
        lmlo: torch.Tensor,
        rcc: torch.Tensor,
        rmlo: torch.Tensor,
    ) -> torch.Tensor:
        all_feats = self.backbone(torch.cat([lcc, lmlo, rcc, rmlo], dim=0))
        f_lcc, f_lmlo, f_rcc, f_rmlo = all_feats.chunk(4, dim=0)

        f_lcc, f_lmlo = self.lateral_left(f_lcc, f_lmlo)
        f_rcc, f_rmlo = self.lateral_right(f_rcc, f_rmlo)

        f_lcc, f_rcc = self.bilateral_cc(f_lcc, f_rcc)
        f_lmlo, f_rmlo = self.bilateral_mlo(f_lmlo, f_rmlo)

        return self.aggregator([f_lcc, f_lmlo, f_rcc, f_rmlo])