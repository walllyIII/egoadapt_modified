# egoadapt/models/policy_pi_avloc_ba.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetASL_BA(nn.Module):
    """
    pi for ASL/BA: same gating idea but with extra locality token to bias spatial/temporal focus.
    Returns gates similar to AR, but you can read out the locality token for debugging.
    """
    def __init__(self, d_feat=256, hidden=256, n_modalities=3, audio_channels: int = 1):
        super().__init__()
        self.local_token = nn.Parameter(torch.randn(1, 1, d_feat))
        self.encoder = nn.GRU(d_feat, hidden, batch_first=True, bidirectional=False)
        self.heads = nn.ModuleList([nn.Linear(hidden, 2) for _ in range(n_modalities + (audio_channels - 1))])

    @staticmethod
    def _gumbel_softmax_sample(logits: torch.Tensor, tau: float, hard: bool):
        g = -torch.empty_like(logits).exponential_().log()
        y = F.softmax((logits + g) / tau, dim=-1)
        if hard:
            idx = y.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
            y = (y_hard - y).detach() + y
        return y, F.softmax(logits, dim=-1)

    def forward(self, feat_seq: torch.Tensor, tau: float = 1.0, hard: bool = True):
        B, T, D = feat_seq.shape
        lt = self.local_token.expand(B, -1, -1)
        x = torch.cat([lt, feat_seq], dim=1)  # [B,1+T,D]
        h, _ = self.encoder(x)                 # [B,1+T,H]
        h = h[:, 1:]                           # drop locality token for gating
        gates_soft, gates_hard = [], []
        for head in self.heads:
            logits = head(h)
            y_relaxed, y_probs = self._gumbel_softmax_sample(logits, tau, hard)
            keep_prob = y_relaxed[..., 1:2]
            gates_soft.append(y_probs)
            gates_hard.append((keep_prob > 0.5).float())
        return gates_soft, gates_hard