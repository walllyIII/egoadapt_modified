# egoadapt/utils/ckpt.py
import torch

def save_ckpt(path, **objects):
    torch.save(objects, path)

def load_ckpt(path):
    return torch.load(path, map_location="cpu")