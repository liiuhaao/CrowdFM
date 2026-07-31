from __future__ import annotations

import json
from pathlib import Path

import dlwheel
import torch

from cfm.data import load_data
from cfm.model.CrowdSIFM import CrowdSIFM
from cfm.si.pipeline import CrowdSIPipelineConfig, run_crowdsi


def _load_checkpoint(model: CrowdSIFM, path: str, device: str) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    if not model.crowdsi_trained:
        raise RuntimeError("checkpoint does not mark CrowdSI heads as trained")


def main() -> None:
    cfg = dlwheel.setup()
    model = CrowdSIFM(**cfg.model.to_dict()).to(cfg.device)
    _load_checkpoint(model, cfg.checkpoint_path, cfg.device)
    model.eval()
    pipeline_config = CrowdSIPipelineConfig(**dict(cfg.crowdsi.to_dict()))

    datasets = load_data.run(cfg)
    results: dict[str, dict] = {}
    for index, (name, data) in enumerate(datasets.items()):
        data.to(cfg.device)
        result = run_crowdsi(
            model,
            data,
            config=pipeline_config,
            seed=int(cfg.get("base_seed", 42000)) + index,
        )
        record = {
            "e_value": result["e_value"],
            "log_e_value": result["log_e_value"],
            "e_value_threshold": result["e_value_threshold"],
            "adaptation_supported": result["adaptation_supported"],
            "used_adaptation": result["used_adaptation"],
        }
        if "accuracy" in result:
            record.update(
                {
                    "adapted_accuracy": result["accuracy"],
                    "zero_shot_accuracy": result["zero_shot_accuracy"],
                    "crowdfm_accuracy": result["crowdfm_accuracy"],
                    "adaptation_delta": (
                        result["accuracy"] - result["zero_shot_accuracy"]
                    ),
                    "crowdsi_zero_shot_delta": (
                        result["zero_shot_accuracy"] - result["crowdfm_accuracy"]
                    ),
                }
            )
        results[name] = record
        accuracy_text = (
            f", adapted={record['adapted_accuracy']:.4f}, "
            f"zero={record['zero_shot_accuracy']:.4f}, "
            f"crowdfm={record['crowdfm_accuracy']:.4f}"
            if "adapted_accuracy" in record
            else ""
        )
        print(
            f"[{name}] e={record['e_value']:.4g}, "
            f"adapt={record['used_adaptation']}{accuracy_text}"
        )

    output_path = Path(cfg.get("output_path", "log/crowdsi_evaluation.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    temporary.replace(output_path)
    print(f"Saved CrowdSI evaluation to {output_path}")


if __name__ == "__main__":
    main()
