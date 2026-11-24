import argparse
import json
import time
from pathlib import Path

import torch

from cfm.checkpoints import read_checkpoint
from cfm.config import prepare_config
from cfm.tasks import available_tasks, dataset_paths, get_task
from cfm.utils import evaluate_scenes, mean_metrics, prepare_evaluation_scene, set_seed


def evaluate(config, checkpoint_path, synthetic=False):
    task = get_task(config["task"])
    checkpoint = read_checkpoint(checkpoint_path, task.name)
    config = dict(config)
    config["model"] = checkpoint["config"]["model"]
    config["simulator"] = checkpoint["config"]["simulator"]
    paths = {} if synthetic else dataset_paths(config["path"]["data"])
    result = {}
    for name, path in (paths.items() if paths else [("synthetic", None)]):
        by_seed = {}
        for seed in config["evaluation"]["seeds"]:
            set_seed(seed)
            data = task.load(config, path) if path else None
            model = task.build_model(config, config["device"])
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            if data is not None:
                data.to(config["device"])
                if str(config["device"]).startswith("cuda"):
                    torch.cuda.synchronize(config["device"])
                started = time.perf_counter()
                values = model.evaluate(data)
                if str(config["device"]).startswith("cuda"):
                    torch.cuda.synchronize(config["device"])
                values["runtime"] = time.perf_counter() - started
            else:
                simulator = task.build_simulator(config)
                scenes = [
                    prepare_evaluation_scene(simulator.generate(), config)
                    for _ in range(config["evaluation"]["num_scenes"])
                ]
                values = evaluate_scenes(model, scenes, config["device"])
            by_seed[str(seed)] = values
        by_seed["avg"] = mean_metrics(list(by_seed.values()))
        result[name] = by_seed
        print(name, json.dumps(by_seed["avg"]))
    result["Average"] = mean_metrics([values["avg"] for values in result.values()])
    result["metadata"] = {
        "task": task.name,
        "checkpoint": str(checkpoint_path),
        "config": config,
        "aggregation": "equal_dataset_then_equal_seed" if paths else "equal_scene_then_equal_seed",
        "runtime_definition": "real-dataset inference including metrics, excluding loading and construction",
    }
    output = Path(config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="CrowdFM evaluate")
    parser.add_argument("--task", choices=available_tasks())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-scenes", type=int)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if not args.task and not args.config:
        parser.error("provide --task or --config")
    return args


def main():
    args = parse_args()
    config = prepare_config(args, "evaluate")
    evaluate(config, args.checkpoint, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
