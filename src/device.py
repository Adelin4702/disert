"""Device selection that works on CUDA, Apple Silicon (MPS), and CPU."""
import torch


def get_device(preferred: str = "auto") -> torch.device:
    """Return the best available device.

    preferred: "auto" | "cuda" | "mps" | "cpu". "auto" picks cuda > mps > cpu.
    """
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_supports_amp(device: torch.device) -> bool:
    """Automatic mixed precision is only equivalent to CUDA AMP on CUDA.

    On MPS, torch.cuda.amp does not apply and behaviour differs, so we run fp32
    there. This is disclosed in the thesis: timing/throughput come from CUDA.
    """
    return device.type == "cuda"


def peak_memory_mb(device: torch.device):
    """Peak allocated memory in MB, or None if the backend doesn't report it."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1e6
        except Exception:
            return None
    return None


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
