"""Ablation variants for CVAF-SwinB.

Four configurations isolate the contribution of each fusion stage:

  SingleView     Swin-B on L-CC only, no cross-view fusion.
  LateralOnly    Stage 1 + Stage 2 (CC<->MLO lateral fusion).
  BilateralOnly  Stage 1 + Stage 3 (L<->R bilateral fusion).
  FullCVAF       Stage 1 + Stage 2 + Stage 3 (the complete model).

All variants share the same backbone, aggregator and training settings, so
any difference in performance is attributable to the fusion design alone.
"""

from typing import List

import torch
import torch.nn as nn
from timm import create_model


class FFN(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expand), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * expand, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SwinBackbone(nn.Module):
    def __init__(self, pretrained=True, img_size=224):
        super().__init__()
        self.swin = create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained, num_classes=0,
            global_pool="", img_size=img_size,
        )
        self.feature_dim = self.swin.num_features

    def forward(self, x):
        out = self.swin.forward_features(x)
        if out.dim() == 4:
            B, H, W, C = out.shape
            out = out.view(B, H * W, C)
        return out


class LateralFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
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

    def forward(self, cc, mlo):
        cc_att, _ = self.cc_to_mlo(query=cc, key=mlo, value=mlo)
        cc = self.norm_cc1(cc + cc_att)
        cc = self.norm_cc2(cc + self.ffn_cc(cc))
        mlo_att, _ = self.mlo_to_cc(query=mlo, key=cc, value=cc)
        mlo = self.norm_mlo1(mlo + mlo_att)
        mlo = self.norm_mlo2(mlo + self.ffn_mlo(mlo))
        return cc, mlo


class BilateralFusion(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
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

    def forward(self, left, right):
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
    def __init__(self, dim, num_views=4, num_classes=1, dropout=0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_views, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, num_views), nn.Softmax(dim=-1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim // 2, num_classes),
        )

    def forward(self, feats: List[torch.Tensor]):
        pooled = [f.mean(dim=1) for f in feats]
        concat = torch.cat(pooled, dim=-1)
        gates = self.gate(concat)
        stacked = torch.stack(pooled, dim=1)
        fused = (gates.unsqueeze(-1) * stacked).sum(dim=1)
        return self.head(fused)


class SingleViewSwinB(nn.Module):
    """Baseline: Swin-B on the L-CC view only."""

    def __init__(self, num_classes=1, img_size=224, pretrained=True,
                 head_dropout=0.3):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=pretrained, img_size=img_size)
        C = self.backbone.feature_dim
        self.aggregator = GlobalAggregator(
            C, num_views=1, num_classes=num_classes, dropout=head_dropout)

    def forward(self, lcc, lmlo, rcc, rmlo):
        return self.aggregator([self.backbone(lcc)])


class LateralOnlySwinB(nn.Module):
    """Stage 1 + Stage 2 only (no bilateral fusion)."""

    def __init__(self, num_classes=1, img_size=224, pretrained=True,
                 num_heads=8, dropout=0.1, head_dropout=0.3):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=pretrained, img_size=img_size)
        C = self.backbone.feature_dim
        self.lateral_left = LateralFusion(C, num_heads, dropout)
        self.lateral_right = LateralFusion(C, num_heads, dropout)
        self.aggregator = GlobalAggregator(
            C, num_views=4, num_classes=num_classes, dropout=head_dropout)

    def forward(self, lcc, lmlo, rcc, rmlo):
        all_f = self.backbone(torch.cat([lcc, lmlo, rcc, rmlo], dim=0))
        f_lcc, f_lmlo, f_rcc, f_rmlo = all_f.chunk(4, dim=0)
        f_lcc, f_lmlo = self.lateral_left(f_lcc, f_lmlo)
        f_rcc, f_rmlo = self.lateral_right(f_rcc, f_rmlo)
        return self.aggregator([f_lcc, f_lmlo, f_rcc, f_rmlo])


class BilateralOnlySwinB(nn.Module):
    """Stage 1 + Stage 3 only (no lateral fusion)."""

    def __init__(self, num_classes=1, img_size=224, pretrained=True,
                 num_heads=8, dropout=0.1, head_dropout=0.3):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=pretrained, img_size=img_size)
        C = self.backbone.feature_dim
        self.bilateral_cc = BilateralFusion(C, num_heads, dropout)
        self.bilateral_mlo = BilateralFusion(C, num_heads, dropout)
        self.aggregator = GlobalAggregator(
            C, num_views=4, num_classes=num_classes, dropout=head_dropout)

    def forward(self, lcc, lmlo, rcc, rmlo):
        all_f = self.backbone(torch.cat([lcc, lmlo, rcc, rmlo], dim=0))
        f_lcc, f_lmlo, f_rcc, f_rmlo = all_f.chunk(4, dim=0)
        f_lcc, f_rcc = self.bilateral_cc(f_lcc, f_rcc)
        f_lmlo, f_rmlo = self.bilateral_mlo(f_lmlo, f_rmlo)
        return self.aggregator([f_lcc, f_lmlo, f_rcc, f_rmlo])


class CVAFSwinB(nn.Module):
    """Complete model: Stage 1 + Stage 2 + Stage 3."""

    def __init__(self, num_classes=1, img_size=224, pretrained=True,
                 num_heads=8, dropout=0.1, head_dropout=0.3):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=pretrained, img_size=img_size)
        C = self.backbone.feature_dim
        self.lateral_left = LateralFusion(C, num_heads, dropout)
        self.lateral_right = LateralFusion(C, num_heads, dropout)
        self.bilateral_cc = BilateralFusion(C, num_heads, dropout)
        self.bilateral_mlo = BilateralFusion(C, num_heads, dropout)
        self.aggregator = GlobalAggregator(
            C, num_views=4, num_classes=num_classes, dropout=head_dropout)

    def forward(self, lcc, lmlo, rcc, rmlo):
        all_f = self.backbone(torch.cat([lcc, lmlo, rcc, rmlo], dim=0))
        f_lcc, f_lmlo, f_rcc, f_rmlo = all_f.chunk(4, dim=0)
        f_lcc, f_lmlo = self.lateral_left(f_lcc, f_lmlo)
        f_rcc, f_rmlo = self.lateral_right(f_rcc, f_rmlo)
        f_lcc, f_rcc = self.bilateral_cc(f_lcc, f_rcc)
        f_lmlo, f_rmlo = self.bilateral_mlo(f_lmlo, f_rmlo)
        return self.aggregator([f_lcc, f_lmlo, f_rcc, f_rmlo])


ABLATION_CONFIGS = {
    'SingleView': SingleViewSwinB,
    'LateralOnly': LateralOnlySwinB,
    'BilateralOnly': BilateralOnlySwinB,
    'FullCVAF': CVAFSwinB,
}


def build_ablation_model(config: str, **kwargs) -> nn.Module:
    """Build an ablation model by name. SingleView ignores num_heads/dropout."""
    if config not in ABLATION_CONFIGS:
        raise ValueError(
            f"Unknown config '{config}'. Choose from: {list(ABLATION_CONFIGS)}")
    return ABLATION_CONFIGS[config](**kwargs)