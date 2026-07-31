from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import dlwheel
import torch

from cfm.audit import CBRConfig
from cfm.audit.calibration import CALIBRATION_VARIANTS, run_calibration_world
from cfm.audit.synthetic import SyntheticWorldConfig, generate_synthetic_world
from cfm.model.CFM import CFM
from cfm.utils import set_seed


DEFAULT_CONFIGURATIONS = [
    {
        "name": "small_binary",
        "num_worker": 20,
        "num_task": 200,
        "num_option": 2,
        "labels_per_task": 5,
    },
    {
        "name": "medium_multiclass",
        "num_worker": 50,
        "num_task": 500,
        "num_option": 5,
        "labels_per_task": 5,
    },
    {
        "name": "large_multiclass",
        "num_worker": 100,
        "num_task": 1000,
        "num_option": 10,
        "labels_per_task": 10,
    },
    {
        "name": "sparse_imbalanced",
        "num_worker": 50,
        "num_task": 500,
        "num_option": 5,
        "labels_per_task": 3,
        "class_imbalance": 1.0,
    },
]


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _normalise_variants(raw_variants: Any) -> list[str]:
    raw_variants = _plain(raw_variants)
    if isinstance(raw_variants, str):
        text = raw_variants.strip()
        try:
            raw_variants = json.loads(text)
        except json.JSONDecodeError:
            raw_variants = [part.strip() for part in text.split(",") if part.strip()]
    variants = list(raw_variants)
    unknown = sorted(set(variants) - set(CALIBRATION_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown calibration variants: {unknown}")
    if not variants:
        raise ValueError("At least one calibration variant is required")
    return variants


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)


def _mean(values: list[dict[str, Any]], key: str) -> float:
    return sum(float(value[key]) for value in values) / len(values)


def _summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (run["configuration"], run["variant"])
        groups.setdefault(key, []).append(run)

    summary: dict[str, Any] = {}
    for (configuration, variant), values in sorted(groups.items()):
        p_values = [float(value["p_value"]) for value in values]
        summary.setdefault(configuration, {})[variant] = {
            "num_worlds": len(values),
            "rejection_rate": sum(bool(value["reject"]) for value in values) / len(values),
            "mean_p_value": sum(p_values) / len(p_values),
            "mean_statistic": _mean(values, "statistic"),
            "mean_audit_posterior_accuracy": _mean(values, "audit_posterior_accuracy"),
            "mean_confusion_mae": _mean(values, "confusion_mae"),
            "mean_disagreement_mae": _mean(values, "disagreement_mae"),
            "mean_nuisance_support": _mean(values, "mean_nuisance_support"),
            "mean_median_nuisance_support": _mean(values, "median_nuisance_support"),
            "mean_min_nuisance_support": _mean(values, "min_nuisance_support"),
            "p_values": p_values,
        }
    return summary


def _write_results(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, output_path)


def main() -> None:
    cfg = dlwheel.setup()
    raw_cbr = _plain(cfg.get("cbr", {}))
    cbr_config = CBRConfig(**dict(raw_cbr))
    raw_calibration = dict(_plain(cfg.get("calibration", {})))
    num_worlds = int(raw_calibration.get("num_worlds", 20))
    base_seed = int(raw_calibration.get("base_seed", 42000))
    variants = _normalise_variants(
        raw_calibration.get("variants", list(CALIBRATION_VARIANTS))
    )
    raw_configurations = _plain(
        raw_calibration.get("configurations", DEFAULT_CONFIGURATIONS)
    )
    configurations = [
        SyntheticWorldConfig(**dict(_plain(configuration)))
        for configuration in raw_configurations
    ]
    if num_worlds < 1:
        raise ValueError("calibration.num_worlds must be positive")

    checkpoint_path = cfg.get("checkpoint_path", "checkpoint.pt")
    output_path = cfg.get("output_path", "log/cbr_calibration_smoke.json")
    model = CFM(**cfg.model.to_dict()).to(cfg.device)
    _load_checkpoint(model, checkpoint_path, cfg.device)
    model.eval()

    payload: dict[str, Any] = {
        "git_commit": _git_commit(),
        "checkpoint_path": checkpoint_path,
        "device": cfg.device,
        "cbr": asdict(cbr_config),
        "calibration": {
            "num_worlds": num_worlds,
            "base_seed": base_seed,
            "variants": variants,
            "configurations": [asdict(configuration) for configuration in configurations],
        },
        "runs": [],
        "summary": {},
    }

    for configuration_index, configuration in enumerate(configurations):
        for world_index in range(num_worlds):
            world_seed = base_seed + configuration_index * 100_000 + world_index
            set_seed(world_seed)
            world = generate_synthetic_world(configuration, seed=world_seed)
            world.data.to(cfg.device)
            audit_seed = world_seed + 10_000
            for variant in variants:
                start = time.perf_counter()
                result = run_calibration_world(
                    model,
                    world,
                    variant=variant,
                    config=cbr_config,
                    seed=audit_seed,
                )
                payload["runs"].append(
                    {
                        "configuration": configuration.name,
                        "world_index": world_index,
                        "world_seed": world_seed,
                        "audit_seed": audit_seed,
                        **asdict(result),
                        "runtime_seconds": time.perf_counter() - start,
                    }
                )
            payload["summary"] = _summarise(payload["runs"])
            _write_results(output_path, payload)
            current = payload["summary"][configuration.name]
            rates = ", ".join(
                f"{variant}={current[variant]['rejection_rate']:.3f}"
                for variant in variants
            )
            print(
                f"[{configuration.name}] {world_index + 1}/{num_worlds}: {rates}",
                flush=True,
            )

    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved calibration results to {output_path}")


if __name__ == "__main__":
    main()
