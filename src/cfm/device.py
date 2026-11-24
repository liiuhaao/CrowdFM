import torch


def validate_device(requested):
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid device {requested!r}") from exc
    if device.type == "cpu":
        return "cpu"
    if device.type != "cuda":
        raise ValueError("device must be cpu or cuda[:index]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is unavailable")
    return str(device)
