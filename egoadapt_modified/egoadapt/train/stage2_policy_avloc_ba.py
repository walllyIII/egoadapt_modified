# egoadapt/train/stage2_policy_avloc_ba.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from egoadapt.models.policy_pi_avloc_ba import PolicyNetASL_BA
from egoadapt.losses.policy_loss import policy_loss


def train_step_policy_avloc_ba(model_phi, model_pi: PolicyNetASL_BA, batch_seq, lambdas, gamma_miscls=1.0, tau=1.0, opt=None, task="asl"):
    model_phi.eval()
    model_pi.train()

    I_seq, A_seq, B_seq = batch_seq["I_seq"], batch_seq["A_seq"], batch_seq["B_seq"]
    Bsz, T = I_seq.shape[:2]

    zs, preds = [], []
    # with torch.no_grad():
    for t in range(T):
        out_t = model_phi(I_seq, A_seq, B_seq)
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

    # if task == "asl":
    #     L_cls = F.cross_entropy(agg_logits, batch_seq["y_asl"])   # active speaker labels
    # else:
    #     L_cls = F.cross_entropy(agg_logits, batch_seq["y_ba"])    # behavior anticipation labels
    # L_eff = 
    # L_pi  = gamma_miscls * L_cls + L_eff
    if task== "asl":
        L_cls,L_eff,L_pi=policy_loss(agg_logits,batch_seq["y_asl"],gates_hard, lambdas)
    else:
        L_cls,L_eff,L_pi=policy_loss(agg_logits,batch_seq["y_ba"],gates_hard, lambdas)
    opt.zero_grad(set_to_none=True)
    L_pi.backward()
    opt.step()

    return {"L_cls": float(L_cls.detach().cpu()), "L_eff": float(L_eff.detach().cpu()), "L_pi": float(L_pi.detach().cpu())}