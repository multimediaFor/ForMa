import torch
import os


def get_device(cuda_idx):
    cuda_device = os.getenv("CUDA_VISIBLE_DEVICES", str(cuda_idx))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_device}")
    else:
        device = torch.device("cpu")

    return device