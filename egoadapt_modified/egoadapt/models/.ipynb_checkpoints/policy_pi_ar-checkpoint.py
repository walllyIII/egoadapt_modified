# egoadapt/models/policy_pi_ar.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetAR(nn.Module):
    """
    pi for Action Recognition (AR): learns keep/drop gates per modality per timestep.
    Collapses audio channels internally if needed.
    """
    def __init__(self, d_feat=256, n_modalities=3, hidden=256, audio_channels: int = 1):
        super().__init__()
        self.n_modalities = n_modalities
        self.audio_channels = audio_channels
        self.lstm = nn.LSTM(input_size=d_feat, hidden_size=hidden, batch_first=True)
        # Gate heads: V (1) + A_ch* + B (1)
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
        h, _ = self.lstm(feat_seq)  # [B,T,H]
        gates_soft, gates_hard = [], []
        for head in self.heads:
            logits = head(h)  # [B,T,2]
            y_relaxed, y_probs = self._gumbel_softmax_sample(logits, tau, hard=hard)
            keep_prob = y_relaxed[..., 1:2]
            gates_soft.append(y_probs)
            gates_hard.append((keep_prob > 0.5).float())
        return gates_soft, gates_hard