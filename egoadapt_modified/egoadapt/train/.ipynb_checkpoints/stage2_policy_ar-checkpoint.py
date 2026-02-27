# egoadapt/train/stage2_policy_ar.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from egoadapt.models.policy_pi_ar import PolicyNetAR
from egoadapt.losses.distillation import policy_efficiency_cost


def train_step_policy_ar(model_phi, model_pi: PolicyNetAR, batch_seq, lambdas, gamma_miscls=1.0, tau=1.0, opt=None):
    model_phi.eval()   # phi frozen in stage 2
    model_pi.train()

    I_seq, A_seq, B_seq = batch_seq["I_seq"], batch_seq["A_seq"], batch_seq["B_seq"]
    Bsz, T = I_seq.shape[:2]

    zs, preds = [], []
    with torch.no_grad():
        for t in range(T):
            out_t = model_phi(I_seq[:, t], A_seq[:, t], B_seq[:, t])
            zs.append(out_t["z_phi"])     # [B,D]
            preds.append(out_t["logits"]) # [B,C]
    z_seq = torch.stack(zs, dim=1)
    logits_seq = torch.stack(preds, dim=1)

    gates_soft, gates_hard = model_pi(z_seq, tau=tau, hard=True)
    keep_any = torch.zeros(Bsz, T, 1, device=z_seq.device)
    for gh in gates_hard:
        keep_any = torch.maximum(keep_any, gh)
    keep_any = keep_any.squeeze(-1)

    masked_logits = logits_seq * keep_any.unsqueeze(-1)
    denom = keep_any.sum(dim=1, keepdim=True).clamp_min(1.0)
    agg_logits = masked_logits.sum(dim=1) / denom

    L_cls = F.cross_entropy(agg_logits, batch_seq["y"])
    L_eff = policy_efficiency_cost(gates_hard, lambdas)
    L_pi  = gamma_miscls * L_cls + L_eff

    opt.zero_grad(set_to_none=True)
    L_pi.backward()
    opt.step()

    return {"L_cls": float(L_cls.detach().cpu()), "L_eff": float(L_eff.detach().cpu()), "L_pi": float(L_pi.detach().cpu())}