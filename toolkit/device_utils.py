import torch
import gc
import os

def get_device() -> torch.device:
    """
    Returns the best available device.
    Prioritizes MPS on macOS, then CUDA, then CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def is_mps_available() -> bool:
    return torch.backends.mps.is_available()

def is_cuda_available() -> bool:
    return torch.cuda.is_available()

def empty_cache():
    """
    Empties the cache for the current device.
    """
    gc.collect()
    if is_mps_available():
        torch.mps.empty_cache()
    elif is_cuda_available():
        torch.cuda.empty_cache()

def manual_seed(seed: int):
    """
    Sets the seed for the current device.
    """
    torch.manual_seed(seed)
    if is_mps_available():
        torch.mps.manual_seed(seed)
    elif is_cuda_available():
        torch.cuda.manual_seed(seed)

def get_device_name() -> str:
    if is_mps_available():
        return "mps"
    elif is_cuda_available():
        return "cuda"
    else:
        return "cpu"

def autocast():
    if is_mps_available():
        return torch.autocast(device_type="mps")
    elif is_cuda_available():
        return torch.autocast(device_type="cuda")
    else:
        # Fallback to cpu or simple context manager
        return torch.autocast(device_type="cpu")
