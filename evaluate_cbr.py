import json
import os
from dataclasses import asdict
from pprint import pprint

import dlwheel
import torch

from cfm.audit import CBRConfig, run_cbr_audit
from cfm.data import load_data
from cfm.model.CFM import CFM
from cfm.utils import normalize_seeds, set_seed


def load_checkpoint(cfg, model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)


def main():
    cfg = dlwheel.setup()
    checkpoint_path = cfg.get("checkpoint_path", "checkpoint.pt")
    output_path = cfg.get("output_path", "log/cbr_audit.json")
    seeds = normalize_seeds(cfg.get("seeds", [42, 43, 44, 45, 46]))
    raw_cbr = cfg.get("cbr", {})
    cbr_kwargs = raw_cbr.to_dict() if hasattr(raw_cbr, "to_dict") else dict(raw_cbr)
    audit_cfg = CBRConfig(**cbr_kwargs)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = {"config": asdict(audit_cfg), "seeds": seeds, "datasets": {}}
    for dataset_name in sorted(load_data.get_dataset_list(cfg)):
        results["datasets"][dataset_name] = {}
        for seed in seeds:
            set_seed(seed)
            data = list(load_data.run(cfg, dataset_name).values())[0].to(cfg.device)
            model = CFM(**cfg.model.to_dict()).to(cfg.device)
            load_checkpoint(cfg, model, checkpoint_path)
            audit = run_cbr_audit(model, data, config=audit_cfg, seed=seed)
            results["datasets"][dataset_name][str(seed)] = {
                "statistic": audit["statistic"],
                "p_value": audit["p_value"],
                "reject": audit["reject"],
                "num_supported_pairs": audit["num_supported_pairs"],
                "num_audit_edges": audit["num_audit_edges"],
            }
        pprint({dataset_name: results["datasets"][dataset_name]})

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
