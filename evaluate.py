import json
import os
import time
from pprint import pformat, pprint

import dlwheel
import torch
from tqdm import tqdm

from cfm.data import load_data
from cfm.model.CFM import CFM
from cfm.utils import set_seed


def load_checkpoint(cfg, model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint, strict=False)


def main():
    cfg = dlwheel.setup()
    output_path = cfg.get("output_path", "log/perform.json")
    checkpoint_path = cfg.get("checkpoint_path", "checkpoint.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seeds = cfg.get("seeds", [42, 43, 44, 45, 46])
    pprint(cfg)

    test_dataset_list = load_data.get_dataset_list(cfg)
    print(test_dataset_list)

    perform = dict()
    for dataset_name in sorted(test_dataset_list):
        perform[dataset_name] = dict()
        avg_perform = dict()
        for seed in seeds:
            set_seed(seed)

            start_time = time.time()
            data = list(load_data.run(cfg, dataset_name).values())[0].to(cfg.device)
            model = CFM(**cfg.model.to_dict()).to(cfg.device)
            load_checkpoint(cfg, model, checkpoint_path)
            perform[dataset_name][seed] = model.evaluate(data)
            end_time = time.time()
            perform[dataset_name][seed]["runtime"] = end_time - start_time
            for k, v in perform[dataset_name][seed].items():
                if k not in avg_perform:
                    avg_perform[k] = []
                avg_perform[k].append(v)

        for k, v in avg_perform.items():
            avg_perform[k] = sum(v) / len(v)
        perform[dataset_name]["avg"] = avg_perform
        acc = perform[dataset_name]["avg"]["task_y_acc"]
        run_time = perform[dataset_name]["avg"]["runtime"]
        print(dataset_name, acc, run_time)

    avg_perform = dict()
    for dataset_name in sorted(test_dataset_list):
        for k, v in perform[dataset_name]["avg"].items():
            if k not in avg_perform:
                avg_perform[k] = []
            avg_perform[k].append(v)
    for k, v in avg_perform.items():
        avg_perform[k] = sum(v) / len(v)
    perform["Average"] = avg_perform
    tqdm.write(f"{pformat(perform['Average'])}")

    json.dump(perform, open(output_path, "w"))


if __name__ == "__main__":
    main()
