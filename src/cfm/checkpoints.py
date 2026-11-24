import json
from pathlib import Path

import torch

SCHEMA_VERSION = 1


def read_checkpoint(path, task, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"schema_version", "task", "iteration", "config", "model", "optimizer"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("checkpoint does not use the CrowdFM checkpoint schema")
    if checkpoint["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if checkpoint["task"] != task or checkpoint["config"]["task"] != task:
        raise ValueError("checkpoint task does not match the requested task")
    return checkpoint


def resolve_resume(path, log_path):
    if path == "latest":
        info = Path(log_path) / "checkpoints" / "latest.json"
        candidate = Path(json.loads(info.read_text())["checkpoint"])
    else:
        candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {candidate}")
    return candidate


def save_checkpoint(path, iteration, model, optimizer, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": SCHEMA_VERSION, "task": config["task"],
        "iteration": iteration, "config": config,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    (path.parent / "latest.json").write_text(json.dumps({"checkpoint": str(path.resolve())}))
