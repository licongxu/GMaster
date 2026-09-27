"""CUDA device-kind detection.

Decides whether JAX is running on an NVIDIA CUDA GPU, which selects the fused
GPU transform paths over the generic fallback. ``device_kind`` strings vary by
platform: workstation drivers report e.g. ``NVIDIA RTX ...``, while Colab reports
``Tesla T4`` / ``Tesla L4`` with no ``NVIDIA`` substring, so the check matches a
list of known product tags rather than the vendor name alone.
"""

import jax

_CUDA_KIND_TAGS = (
    "NVIDIA",
    "TESLA",
    "GEFORCE",
    "QUADRO",
    "TITAN",
    "RTX",
    "A100",
    "A10G",
    "H100",
    "H200",
    "B200",
    "L4",
    "L40",
    "T4",
    "V100",
    "P100",
    "P40",
    "K80",
    "CUDA",
)


def is_cuda_device_kind(kind):
    """True for JAX ``device_kind`` strings that name an NVIDIA CUDA GPU."""
    text = (kind or "").upper()
    if not text or text in {"GPU", "CUDA"}:
        return True
    return any(tag in text for tag in _CUDA_KIND_TAGS)


def on_cuda_gpu(devices=None):
    """True when JAX is running on a CUDA GPU, including Colab's Tesla T4."""
    devices = jax.devices() if devices is None else devices
    return any(
        device.platform == "gpu"
        and is_cuda_device_kind(getattr(device, "device_kind", "") or "GPU")
        for device in devices
    )
