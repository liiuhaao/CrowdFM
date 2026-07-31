from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cfm.data.crowd_data import CrowdData


@dataclass(frozen=True)
class SyntheticWorldConfig:
    name: str
    num_worker: int
    num_task: int
    num_option: int
    labels_per_task: int
    dim: int = 32
    class_imbalance: float = 0.0
    accuracy_alpha: float = 8.0
    accuracy_beta: float = 2.0
    off_diagonal_concentration: float = 1.0


@dataclass(frozen=True)
class SyntheticWorld:
    data: CrowdData
    truth: torch.Tensor
    true_confusion: torch.Tensor
    class_prior: torch.Tensor
    config: SyntheticWorldConfig
    seed: int


def _validate_config(config: SyntheticWorldConfig) -> None:
    if config.num_worker < 3:
        raise ValueError("num_worker must be at least three")
    if config.num_task < 2:
        raise ValueError("num_task must be at least two")
    if config.num_option < 2:
        raise ValueError("num_option must be at least two")
    if not 3 <= config.labels_per_task <= config.num_worker:
        raise ValueError("labels_per_task must lie in [3, num_worker]")
    if config.class_imbalance < 0:
        raise ValueError("class_imbalance must be nonnegative")
    if config.accuracy_alpha <= 0 or config.accuracy_beta <= 0:
        raise ValueError("Beta accuracy parameters must be positive")
    if config.off_diagonal_concentration <= 0:
        raise ValueError("off_diagonal_concentration must be positive")


def _class_prior(num_option: int, imbalance: float) -> np.ndarray:
    logits = -imbalance * np.arange(num_option, dtype=np.float64)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def _sample_confusion(
    rng: np.random.Generator,
    config: SyntheticWorldConfig,
) -> np.ndarray:
    num_worker = config.num_worker
    num_option = config.num_option
    chance = 1.0 / num_option
    accuracy = rng.beta(
        config.accuracy_alpha,
        config.accuracy_beta,
        size=(num_worker, num_option),
    )
    accuracy = chance + (1.0 - chance) * accuracy
    accuracy = np.clip(accuracy, chance + 1e-3, 0.995)

    confusion = np.zeros((num_worker, num_option, num_option), dtype=np.float32)
    concentration = np.full(num_option - 1, config.off_diagonal_concentration)
    for worker in range(num_worker):
        for truth in range(num_option):
            off_diagonal = rng.dirichlet(concentration)
            row = np.empty(num_option, dtype=np.float32)
            row[truth] = accuracy[worker, truth]
            row[np.arange(num_option) != truth] = (
                1.0 - accuracy[worker, truth]
            ) * off_diagonal
            confusion[worker, truth] = row
    return confusion


def generate_synthetic_world(
    config: SyntheticWorldConfig,
    *,
    seed: int,
) -> SyntheticWorld:
    """Generate a fixed-degree Dawid--Skene world with known nuisance values."""

    _validate_config(config)
    rng = np.random.default_rng(seed)
    class_prior_np = _class_prior(config.num_option, config.class_imbalance)
    confusion_np = _sample_confusion(rng, config)
    truth_np = rng.choice(
        config.num_option,
        size=config.num_task,
        p=class_prior_np,
    )

    workers: list[int] = []
    answers: list[int] = []
    tasks: list[int] = []
    for task, truth in enumerate(truth_np):
        task_workers = rng.choice(
            config.num_worker,
            size=config.labels_per_task,
            replace=False,
        )
        for worker in task_workers:
            answer = rng.choice(
                config.num_option,
                p=confusion_np[worker, truth],
            )
            workers.append(int(worker))
            answers.append(int(answer))
            tasks.append(task)

    triple = torch.tensor([workers, answers, tasks], dtype=torch.long)
    data = CrowdData(
        dim=config.dim,
        num_worker=config.num_worker,
        num_task=config.num_task,
        num_option=config.num_option,
        triple=triple,
    )
    data.task_y = torch.from_numpy(truth_np).long()
    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    data.setup()
    torch.random.set_rng_state(torch_state)

    return SyntheticWorld(
        data=data,
        truth=data.task_y,
        true_confusion=torch.from_numpy(confusion_np),
        class_prior=torch.from_numpy(class_prior_np).float(),
        config=config,
        seed=seed,
    )
