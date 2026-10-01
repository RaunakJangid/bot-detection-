from __future__ import annotations

import platform

import psutil
import torch


def get_device(cpu: bool = False, smoke: bool = False) -> torch.device:
    """CUDA unless --cpu. Fails loudly when a GPU is expected but missing (smoke runs may use CPU)."""
    if cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if smoke:
        return torch.device("cpu")
    raise RuntimeError(
        "No CUDA GPU found. Install the CUDA build (`uv sync --extra cu128`) or pass --cpu explicitly."
    )


def hardware_record() -> dict:
    rec = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0)
        rec["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return rec
