# egoadapt/utils/optim.py
import torch

def make_opts(model_student, model_policy, lr_student=2e-4, lr_policy=1e-4):
    opt_student = torch.optim.AdamW(model_student.parameters(), lr=lr_student, weight_decay=0.01)
    opt_policy  = torch.optim.AdamW(model_policy.parameters(),  lr=lr_policy,  weight_decay=0.01)
    opt_joint   = torch.optim.AdamW(list(model_student.parameters()) + list(model_policy.parameters()),
                                    lr=min(lr_student, lr_policy), weight_decay=0.01)
    return opt_student, opt_policy, opt_joint