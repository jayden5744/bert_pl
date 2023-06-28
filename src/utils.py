import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        device = "cuda"

    elif torch.backends.mps.is_available():
        device = "mps"

    else:
        device = "cpu"
    return torch.device(device)
