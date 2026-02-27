# egoadapt/train/stage3_joint.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from egoadapt.losses.distillation_loss import distillation_loss, DistillWeights
from egoadapt.losses.policy_loss import policy_loss


def train_step_joint(model_phi, model_pi, batch_seq, teacher_logits_seq, cfd_w: DistillWeights, lambdas, gamma_miscls=1.0, tau=1.0, eta1=1.0, eta2=1.0, opt=None):
    model_phi.train()
    model_pi.train()

    I_seq, A_seq, B_seq = batch_seq["I_seq"], batch_seq["A_seq"], batch_seq["B_seq"]
    Bsz, T = I_seq.shape[:2]

    zs, preds, Lphi_terms = [], [], []
    for t in range(T):
        out_t = model_phi(I_seq, A_seq, B_seq)
        zs.append(out_t["z_phi"])     # [B,D]
        preds.append(out_t["logits"]) # [B,C]
        ldict = distillation_loss(out_t["logits"], teacher_logits_seq, batch_seq["y_asl"], cfd_w)
        Lphi_terms.append(ldict)

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

    # L_cls,L_eff,L_pi = F.cross_entropy(agg_logits, batch_seq["y"])
    # L_eff = policy_loss(gates_hard=gates_hard, lambdas=lambdas)
    # L_pi  = gamma_miscls * L_cls + L_eff

    L_cls,L_eff,L_pi = policy_loss(agg_logits,batch_seq["y_asl"],gates_hard,lambdas)

    print(Lphi_terms[0].keys())
    # Aggregate phi losses
    L_KD = torch.stack([t["L_KD"] for t in Lphi_terms]).mean()
    L_GT = torch.stack([t["L_GT"] for t in Lphi_terms]).mean()
    L_1  = torch.stack([t["L_1"]  for t in Lphi_terms]).mean()
    L_phi = torch.stack([t["L_phi"] for t in Lphi_terms]).mean()

    L_theta = eta1 * L_pi + eta2 * L_phi

    opt.zero_grad(set_to_none=True)
    L_theta.backward()
    opt.step()

    return {
        "L_cls": float(L_cls.detach().cpu()), "L_eff": float(L_eff.detach().cpu()), "L_pi": float(L_pi.detach().cpu()),
        "L_KD": float(L_KD.detach().cpu()), "L_GT": float(L_GT.detach().cpu()), "L_1": float(L_1.detach().cpu()),
        "L_phi": float(L_phi.detach().cpu()), "L_theta": float(L_theta.detach().cpu())
    }