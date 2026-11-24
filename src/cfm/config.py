import copy
import json
from pathlib import Path

from .device import validate_device
from .tasks import get_task


def merge_config(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError("cyclic configuration inheritance")
    seen.add(path)
    current = json.loads(path.read_text())
    base_path = current.pop("extends", None)
    return merge_config(load_config(path.parent / base_path, seen), current) if base_path else current


def apply_overrides(config, overrides):
    config = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"expected key=value override, got {override!r}")
        key, raw = override.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parts = key.split(".")
        if any(not part for part in parts):
            raise ValueError("override key cannot be empty")
        target = config
        for part in parts[:-1]:
            if not isinstance(target.get(part), dict):
                raise ValueError(f"unknown configuration section in {key!r}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"unknown configuration key {key!r}")
        target[parts[-1]] = value
    return config


def prepare_config(args, mode):
    config = load_config(args.config or Path("config") / f"{args.task}.json")
    config = apply_overrides(config, args.set)
    if args.task and config["task"] != args.task:
        raise ValueError("--task conflicts with the configuration")
    if args.device:
        config["device"] = args.device
    if args.name:
        config["name"] = args.name
    if args.data:
        config["path"]["data"] = str(args.data)
    if args.seed is not None:
        if mode == "train":
            config["seed"] = args.seed
        else:
            config["evaluation"]["seeds"] = [args.seed]
    if mode == "train":
        if args.iterations is not None:
            config["iterations"] = args.iterations
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
    else:
        if args.output:
            config["output_path"] = str(args.output)
        if args.num_scenes is not None:
            config["evaluation"]["num_scenes"] = args.num_scenes
    if not config["name"] or not config["output_path"]:
        raise ValueError("configuration requires an experiment name and output path")
    validate_config(config)
    config["device"] = validate_device(config["device"])
    return config


def validate_config(config):
    task = get_task(config.get("task"))
    for key in ("iterations", "batch_size", "gradient_accumulation_steps", "test_interval", "save_interval", "validation_scenes"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["num_workers"] < 0:
        raise ValueError("num_workers cannot be negative")
    for key in ("cpu_threads", "eval_batch_size", "eval_max_edges"):
        if key in config and (not isinstance(config[key], int) or config[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    if config["save_interval"] % config["gradient_accumulation_steps"]:
        raise ValueError("save_interval must be divisible by gradient_accumulation_steps")
    if config["lr"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("invalid optimizer configuration")
    if config["evaluation"]["num_scenes"] < 1 or not config["evaluation"]["seeds"]:
        raise ValueError("evaluation requires scenes and at least one seed")
    if task.validate is not None:
        task.validate(config)
    return config
