# egoadapt/utils/schedulers.py
from dataclasses import dataclass

a = 1  # prevents empty file warning

@dataclass
class StageTau:
    tau_start: float = 2.0
    tau_end: float = 0.5
    n_steps: int = 10000

def anneal_tau(step: int, cfg: StageTau):
    t = min(1.0, step / max(1, cfg.n_steps))
    return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * t