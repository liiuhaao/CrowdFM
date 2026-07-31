from __future__ import annotations

import json
from pathlib import Path

import dlwheel
import torch

from cfm.audit.predictive_pipeline import PredictiveAuditConfig, run_predictive_audit
from cfm.data import load_data
from cfm.model.PredictiveCFM import PredictiveCFM


def _load_predictive_checkpoint(
    model: PredictiveCFM,
    checkpoint_path: str,
    device: str,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    if not model.response_head_trained:
        raise RuntimeError(
            "checkpoint does not mark the response head as trained; refusing to "
            "interpret random response logits as a predictive audit"
        )


def main() -> None:
    cfg = dlwheel.setup()
    audit_config = PredictiveAuditConfig(**dict(cfg.predictive_audit.to_dict()))
    model = PredictiveCFM(**cfg.model.to_dict()).to(cfg.device)
    _load_predictive_checkpoint(model, cfg.checkpoint_path, cfg.device)
    model.eval()

    datasets = load_data.run(cfg)
    results: dict[str, dict] = {}
    for index, (name, data) in enumerate(datasets.items()):
        data.to(cfg.device)
        result = run_predictive_audit(
            model,
            data,
            config=audit_config,
            seed=int(cfg.get("base_seed", 42000)) + index,
        )
        results[name] = {
            "p_value": result["p_value"],
            "marginal_p_value": result["marginal_p_value"],
            "dependence_p_value": result["dependence_p_value"],
            "reject": result["reject"],
            "marginal_statistic": result["marginal_statistic"],
            "dependence_statistic": result["dependence_statistic"],
            "num_supported_pairs": result["num_supported_pairs"],
            "num_audit_edges": result["num_audit_edges"],
            "num_context_edges": result["num_context_edges"],
        }
        print(
            f"[{name}] p={result['p_value']:.4f}, "
            f"marginal={result['marginal_p_value']:.4f}, "
            f"dependence={result['dependence_p_value']:.4f}, "
            f"reject={result['reject']}"
        )

    output_path = Path(cfg.get("output_path", "log/predictive_audit.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    temporary.replace(output_path)
    print(f"Saved predictive audit results to {output_path}")


if __name__ == "__main__":
    main()
