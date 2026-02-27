# egoadapt/models/teachers/swl_teacher_lite.py
"""
SWL (ECCV'24) style teacher for ASL/BA, implemented as a lite, faithful model
when public code is not available. It follows the paper's MuST idea:
- spherical (world-locked) positional embeddings
- modality-wise self-attention
- multi-head decoding for ASL and BA
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SphericalPE(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(3, d)
    def forward(self, unit_vecs: torch.Tensor) -> torch.Tensor:
        # unit_vecs: [B,T,3] (e.g., head orientation / gaze / trajectory)
        return self.lin(unit_vecs)

class ModalityBlock(nn.Module):
    def __init__(self, d: int, nhead: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
    def forward(self, x):
        h,_ = self.self_attn(x, x, x)
        x = self.norm1(x + h)
        h = self.ff(x)
        x = self.norm2(x + h)
        return x

class SWLTeacherLite(nn.Module):
    def __init__(self, d=256, nhead=4, n_layers=4, n_classes_asl=2, n_classes_ba=20):
        super().__init__()
        self.pe = SphericalPE(d)
        self.av_block = nn.ModuleList([ModalityBlock(d, nhead) for _ in range(n_layers)])
        self.beh_block = nn.ModuleList([ModalityBlock(d, nhead) for _ in range(n_layers)])
        self.fuse = nn.Linear(3*d, d)
        self.cls_asl = nn.Linear(d, n_classes_asl)
        self.cls_ba  = nn.Linear(d, n_classes_ba)
    def forward(self, v_tokens, a_tokens, b_dirs):
        # v_tokens, a_tokens: [B,T,D]; b_dirs: [B,T,3] unit vectors (head/gaze/trajectory)
        b_tokens = self.pe(b_dirs)
        for blk in self.av_block:
            v_tokens = blk(v_tokens)
            a_tokens = blk(a_tokens)
        for blk in self.beh_block:
            b_tokens = blk(b_tokens)
        # simple mean pool per stream then fuse
        vz = v_tokens.mean(dim=1)
        az = a_tokens.mean(dim=1)
        bz = b_tokens.mean(dim=1)
        z = torch.cat([vz, az, bz], dim=-1)
        z = self.fuse(z)
        return dict(
            asl_logits=self.cls_asl(z),  # active speaker on/off or multi-class
            ba_logits=self.cls_ba(z),    # behavior anticipation classes
        )