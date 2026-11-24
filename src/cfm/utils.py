import os
import random

import numpy as np
import torch


def set_seed(seed=42):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return seed


def mean_metrics(metrics):
    if not metrics:
        raise ValueError("cannot average an empty metric list")
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in metrics[0]}


def prepare_training_scenes(scenes, config, iteration):
    """Task extension hook applied to every generated training batch."""
    return scenes


def prepare_evaluation_scene(scene, config):
    """Task extension hook applied to every generated evaluation scene."""
    return scene


def _scene_size(scene):
    if hasattr(scene, "edge_index"):
        return scene.edge_index.shape[1]
    if hasattr(scene, "triple"):
        return scene.triple.shape[1]
    return 1


def evaluate_scene_metrics(model, scenes, device, batch_size=16, max_edges=100000):
    if not hasattr(model, "evaluate_batch"):
        results = []
        for scene in scenes:
            try:
                results.append(model.evaluate(scene.to(device)))
            finally:
                scene.to("cpu")
        return results
    results = []
    chunk, edges = [], 0

    def flush():
        try:
            for scene in chunk:
                scene.to(device)
            results.extend(model.evaluate_batch(chunk))
        finally:
            for scene in chunk:
                scene.to("cpu")

    for scene in scenes:
        size = _scene_size(scene)
        if chunk and (len(chunk) >= batch_size or edges + size > max_edges):
            flush()
            chunk, edges = [], 0
        chunk.append(scene)
        edges += size
    if chunk:
        flush()
    return results


def evaluate_scenes(model, scenes, device, **options):
    return mean_metrics(evaluate_scene_metrics(model, scenes, device, **options))


def build_validation_scenes(task, config):
    set_seed(config["seed"] + 1)
    validation_config = dict(config)
    validation_config["simulator"] = dict(config["simulator"])
    validation_config["simulator"].update(config.get("validation_simulator_overrides", {}))
    simulator = task.build_simulator(validation_config)
    return [
        prepare_evaluation_scene(simulator.generate(), config)
        for _ in range(config["validation_scenes"])
    ]
