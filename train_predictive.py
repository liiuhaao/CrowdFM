from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dlwheel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfm.audit.predictive_training import (
    PredictiveTrainingConfig,
    predictive_training_loss,
)
from cfm.data.crowd_dataset import CrowdDataset
from cfm.model.PredictiveCFM import PredictiveCFM


def _load_checkpoint(path: str, device: str) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    return checkpoint


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: PredictiveCFM,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "predictive_training": config,
        },
        path,
    )


def main() -> None:
    cfg = dlwheel.setup()
    training_config = PredictiveTrainingConfig(
        **dict(cfg.predictive_training.to_dict())
    )
    model = PredictiveCFM(**cfg.model.to_dict()).to(cfg.device)

    backbone_checkpoint = cfg.get("backbone_checkpoint_path", None)
    if backbone_checkpoint:
        checkpoint = _load_checkpoint(backbone_checkpoint, cfg.device)
        model.load_crowdfm_checkpoint(checkpoint, strict=True)
    model.freeze_backbone(bool(cfg.get("freeze_backbone", True)))

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters remain")
    optimizer = torch.optim.Adam(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    dataloader = DataLoader(
        CrowdDataset(**cfg.simulator.to_dict()),
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        collate_fn=lambda values: values,
    )
    data_iterator = iter(dataloader)

    output_dir = Path(cfg.get("output_dir", "log/predictive_cfm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_plain(cfg), handle, indent=2)

    accumulation = int(cfg.gradient_accumulation_steps)
    epochs = int(cfg.epochs)
    if accumulation < 1 or epochs < 1:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")

    optimizer.zero_grad()
    model.train()
    with tqdm(range(1, epochs + 1), dynamic_ncols=True) as progress:
        for epoch in progress:
            batch = [data.to(cfg.device) for data in next(data_iterator)]
            losses = []
            annotation_losses = []
            truth_losses = []
            for index, data in enumerate(batch):
                result = predictive_training_loss(
                    model,
                    data,
                    config=training_config,
                    seed=epoch * 1_000_003 + index,
                )
                losses.append(result["loss"])
                annotation_losses.append(result["annotation_loss"].detach())
                truth_losses.append(result["truth_loss"].detach())

            loss = torch.stack(losses).mean() / accumulation
            loss.backward()
            should_step = epoch % accumulation == 0 or epoch == epochs
            if should_step:
                optimizer.step()
                optimizer.zero_grad()
                model.mark_response_head_trained()

            metrics = {
                "loss": float(loss.detach().item() * accumulation),
                "annotation_loss": float(torch.stack(annotation_losses).mean().item()),
                "truth_loss": float(torch.stack(truth_losses).mean().item()),
            }
            progress.set_postfix(metrics)

            if epoch % int(cfg.save_interval) == 0 or epoch == epochs:
                _save_checkpoint(
                    output_dir / f"{epoch}.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    config=training_config.__dict__,
                )


if __name__ == "__main__":
    main()
