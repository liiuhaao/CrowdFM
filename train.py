import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cfm.checkpoints import read_checkpoint, resolve_resume, save_checkpoint
from cfm.config import prepare_config
from cfm.data.dataset import SyntheticDataset, collate_scenes
from cfm.tasks import available_tasks, dataset_paths, get_task
from cfm.utils import (
    build_validation_scenes,
    evaluate_scene_metrics,
    evaluate_scenes,
    prepare_training_scenes,
    set_seed,
)


def learning_rate(config, update):
    base = config["lr"]
    schedule = config.get("lr_schedule")
    if schedule is None:
        return base
    if not isinstance(schedule, dict) or set(schedule) != {"type", "start_update", "end_factor"}:
        raise ValueError("lr_schedule requires type, start_update, end_factor")
    total = math.ceil(config["iterations"] / config["gradient_accumulation_steps"])
    start, factor = schedule["start_update"], schedule["end_factor"]
    if schedule["type"] != "cosine_decay" or type(start) is not int or not 0 <= start < total:
        raise ValueError("invalid cosine decay start_update")
    if not isinstance(factor, (int, float)) or not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("end_factor must be finite and in (0, 1]")
    if not 1 <= update <= total:
        raise ValueError("optimizer update outside training budget")
    progress = max(0, update - start) / (total - start)
    return base * (factor + (1 - factor) * (1 + math.cos(math.pi * progress)) / 2)


def train(config, backup=False, resume=None):
    learning_rate(config, 1)
    if config.get("cpu_threads"):
        torch.set_num_threads(config["cpu_threads"])
    task = get_task(config["task"])
    device = config["device"]
    log_path = Path(config["path"]["log"]) / config["name"]
    start = 1
    checkpoint = None
    if resume:
        checkpoint = read_checkpoint(resolve_resume(resume, log_path), task.name)
        saved = checkpoint["config"]
        if saved["model"] != config["model"] or saved["simulator"] != config["simulator"]:
            raise ValueError("resume model/simulator configuration must match the checkpoint")
        if saved.get("lr_schedule") != config.get("lr_schedule"):
            raise ValueError("resume learning rate schedule must match the checkpoint")
        start = checkpoint["iteration"] + 1
    if start > config["iterations"]:
        raise ValueError("iterations must exceed the checkpoint iteration")
    checkpoint_dir = log_path / "checkpoints"
    if backup and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not resume:
        raise FileExistsError("experiment has checkpoints; choose another name or resume")

    validation = build_validation_scenes(task, config)
    set_seed(config["seed"])
    model = task.build_model(config, device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"], group["weight_decay"] = config["lr"], config["weight_decay"]

    simulator = task.build_simulator(config)
    loader = DataLoader(
        SyntheticDataset(simulator), batch_size=config["batch_size"],
        num_workers=config["num_workers"], collate_fn=collate_scenes,
    )
    batches = iter(loader)
    real_paths = dataset_paths(config["path"]["data"])
    real_data = {name: task.load(config, path) for name, path in real_paths.items()}

    writer = None
    metrics_file = None
    if backup:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("--backup requires tensorboard to be installed") from exc
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (log_path / "config.json").write_text(json.dumps(config, indent=2))
        writer = SummaryWriter(log_path / "run", purge_step=start if resume else None)
        metrics_file = (log_path / "metrics.jsonl").open("a")

    print(json.dumps(config, indent=2))
    accumulation = config["gradient_accumulation_steps"]
    optimizer.zero_grad(set_to_none=True)
    interval_loss, interval_batches = 0.0, 0
    interval_timing = {"data_seconds": 0.0, "train_seconds": 0.0}
    try:
        for iteration in tqdm(range(start, config["iterations"] + 1), desc=task.name):
            model.train()
            started = time.perf_counter()
            scenes = next(batches)
            scenes = prepare_training_scenes(scenes, config, iteration)
            interval_timing["data_seconds"] += time.perf_counter() - started

            started = time.perf_counter()
            scenes = [scene.to(device) for scene in scenes]
            loss = model.batch_loss(scenes)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at iteration {iteration}")
            group_start = start + ((iteration - start) // accumulation) * accumulation
            group_size = min(accumulation, config["iterations"] - group_start + 1)
            (loss / group_size).backward()
            update = (iteration - start + 1) % accumulation == 0 or iteration == config["iterations"]
            if update:
                rate = learning_rate(config, (iteration - 1) // accumulation + 1)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_value = loss.item()
            interval_timing["train_seconds"] += time.perf_counter() - started
            interval_loss += loss_value
            interval_batches += 1
            if writer:
                writer.add_scalar("Train/loss", loss_value, iteration)

            if iteration % config["test_interval"] == 0 or iteration == config["iterations"]:
                model.eval()
                evaluation_started = time.perf_counter()
                evaluation_options = {
                    "batch_size": config.get("eval_batch_size", 16),
                    "max_edges": config.get("eval_max_edges", 100000),
                }
                result = {"synthetic": evaluate_scenes(model, validation, device, **evaluation_options)}
                real_metrics = evaluate_scene_metrics(
                    model, list(real_data.values()), device, **evaluation_options
                ) if real_data else []
                result.update(zip(real_data, real_metrics))
                record = {
                    "iteration": iteration,
                    "training": {
                        "interval_mean_loss": interval_loss / interval_batches,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    "metrics": result,
                    "timing": {
                        **interval_timing,
                        "evaluation_seconds": time.perf_counter() - evaluation_started,
                        "training_batches": interval_batches,
                        "training_scenes": interval_batches * config["batch_size"],
                        "cpu_threads": torch.get_num_threads(),
                    },
                }
                tqdm.write(json.dumps(record, allow_nan=False))
                if metrics_file:
                    metrics_file.write(json.dumps(record, allow_nan=False) + "\n")
                    metrics_file.flush()
                if writer:
                    writer.add_scalar("Train/interval_mean_loss", record["training"]["interval_mean_loss"], iteration)
                    for name, values in result.items():
                        for key, value in values.items():
                            writer.add_scalar(f"{name}/{key}", value, iteration)
                interval_timing = {"data_seconds": 0.0, "train_seconds": 0.0}
                interval_loss, interval_batches = 0.0, 0

            save_due = (iteration - start + 1) % config["save_interval"] == 0
            if backup and update and (save_due or iteration == config["iterations"]):
                save_checkpoint(checkpoint_dir / f"{iteration}.pt", iteration, model, optimizer, config)
    finally:
        if metrics_file:
            metrics_file.close()
        if writer:
            writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="CrowdFM train")
    parser.add_argument("--task", choices=available_tasks())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--backup", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", type=Path)
    resume.add_argument("--resume-latest", action="store_true")
    args = parser.parse_args()
    if not args.task and not args.config:
        parser.error("provide --task or --config")
    return args


def main():
    args = parse_args()
    config = prepare_config(args, "train")
    resume = "latest" if args.resume_latest else args.resume
    train(config, backup=args.backup, resume=resume)


if __name__ == "__main__":
    main()
