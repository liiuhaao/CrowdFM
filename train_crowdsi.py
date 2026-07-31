from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dlwheel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfm.data.crowdsi_simulator import CrowdSIDataset
from cfm.model.CrowdSIFM import CrowdSIFM
from cfm.si.training import CrowdSITrainingConfig, crowdsi_training_loss


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return {key: _plain(item) for key, item in value.to_dict().items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _load_checkpoint(path: str, device: str) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    return checkpoint


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: CrowdSIFM,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "crowdsi_training": config,
        },
        path,
    )


def main() -> None:
    cfg = dlwheel.setup()
    training_config = CrowdSITrainingConfig(**dict(cfg.crowdsi_training.to_dict()))
    model = CrowdSIFM(**cfg.model.to_dict()).to(cfg.device)
    backbone_checkpoint = cfg.get("backbone_checkpoint_path", None)
    if backbone_checkpoint:
        model.load_crowdfm_checkpoint(
            _load_checkpoint(backbone_checkpoint, cfg.device),
            strict=True,
        )
    model.freeze_backbone(bool(cfg.get("freeze_backbone", False)))

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable CrowdSI parameters remain")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    dataloader = DataLoader(
        CrowdSIDataset(**cfg.simulator.to_dict()),
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        collate_fn=lambda values: values,
    )
    data_iterator = iter(dataloader)

    output_dir = Path(cfg.get("output_dir", "log/crowdsi"))
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
            results = [
                crowdsi_training_loss(
                    model,
                    data,
                    config=training_config,
                    seed=epoch * 1_000_003 + index,
                )
                for index, data in enumerate(batch)
            ]
            loss = torch.stack([result["loss"] for result in results]).mean()
            (loss / accumulation).backward()
            should_step = epoch % accumulation == 0 or epoch == epochs
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.gradient_clip))
                optimizer.step()
                optimizer.zero_grad()
                model.mark_crowdsi_trained()

            metric_names = [
                "annotation_loss",
                "truth_loss",
                "assignment_loss",
                "mechanism_kl",
                "consistency_loss",
            ]
            metrics = {"loss": float(loss.detach().item())}
            for name in metric_names:
                metrics[name] = float(
                    torch.stack([result[name].detach() for result in results]).mean().item()
                )
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
