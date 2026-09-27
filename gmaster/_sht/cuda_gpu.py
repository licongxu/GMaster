"""CUDA GPU detection that includes Colab's Tesla T4.

JAX on Colab reports ``device_kind`` as ``Tesla T4`` / ``Tesla L4`` with no
``NVIDIA`` substring. Workstation JAX often reports ``NVIDIA RTX ...``. Requiring
``NVIDIA`` sent the public Colab demo down the generic s2fft scatter loop
(minutes at NSIDE 1024) while the v2 march library still loaded and printed
enabled.
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
