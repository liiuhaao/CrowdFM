import random
from types import SimpleNamespace

import numpy as np
import torch

from ..classification import ClassificationData
from .base import BaseCrowdSimulator


class ClassificationSimulator(BaseCrowdSimulator):
    """Multiclass observations generated from a 3PL worker/task model."""

    def __init__(self, dim=32, num_option_range=(2, 20),
                 ability_mu_range=(-1.0, 1.0), ability_sigma_range=(0.5, 2.0),
                 difficulty_mu_range=(-1.0, 1.0), difficulty_sigma_range=(0.5, 2.0),
                 discrimination_min_range=(0.3, 1.0), discrimination_max_range=(1.5, 3.0),
                 quality_upper_range=(0.1, 0.4), **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_option_range = num_option_range
        if not 2 <= self.num_option_range[0] <= self.num_option_range[1]:
            raise ValueError("classification requires at least two options")
        self.ability_mu_range = tuple(ability_mu_range)
        self.ability_sigma_range = tuple(ability_sigma_range)
        self.difficulty_mu_range = tuple(difficulty_mu_range)
        self.difficulty_sigma_range = tuple(difficulty_sigma_range)
        self.discrimination_min_range = tuple(discrimination_min_range)
        self.discrimination_max_range = tuple(discrimination_max_range)
        self.quality_upper_range = tuple(quality_upper_range)
        for name in ("ability_sigma_range", "difficulty_sigma_range",
                     "discrimination_min_range", "discrimination_max_range"):
            lower, upper = getattr(self, name)
            if lower <= 0 or lower > upper:
                raise ValueError(f"invalid {name}")
        if not 0 <= self.quality_upper_range[0] <= self.quality_upper_range[1] < 1:
            raise ValueError("quality upper range must lie in [0, 1)")
        self.D = 1.7

    def generate(self):
        latent = SimpleNamespace(
            num_worker=random.randint(*self.num_worker_range),
            num_task=random.randint(*self.num_task_range),
            num_option=random.randint(*self.num_option_range),
            num_answer_each_task=random.randint(*self.num_answer_each_task_range),
        )
        truth = torch.randint(0, latent.num_option, (latent.num_task,))
        ability_mu = random.uniform(*self.ability_mu_range)
        ability_sigma = random.uniform(*self.ability_sigma_range)
        worker_ability = torch.randn(latent.num_worker) * ability_sigma + ability_mu
        difficulty_mu = random.uniform(*self.difficulty_mu_range)
        difficulty_sigma = random.uniform(*self.difficulty_sigma_range)
        task_difficulty = torch.randn(latent.num_task) * difficulty_sigma + difficulty_mu
        discrimination_lower, discrimination_upper = sorted((
            random.uniform(*self.discrimination_min_range),
            random.uniform(*self.discrimination_max_range),
        ))
        task_discrimination = torch.rand(latent.num_task) * (
            discrimination_upper - discrimination_lower) + discrimination_lower
        chance = 1.0 / latent.num_option
        quality_upper = random.uniform(*self.quality_upper_range)
        if chance >= quality_upper:
            quality_upper = chance + 0.05
        task_quality_floor = torch.rand(latent.num_task) * (quality_upper - chance) + chance
        workers, tasks = self._sample_graph(latent)
        true_answers = truth.numpy()[tasks]
        theta = worker_ability.numpy()[workers]
        beta = task_difficulty.numpy()[tasks]
        alpha = task_discrimination.numpy()[tasks]
        floor = task_quality_floor.numpy()[tasks]
        logit = self.D * alpha * (theta - beta)
        sigmoid = 1.0 / (1.0 + np.exp(-logit))
        probability = floor + (1.0 - floor) * sigmoid
        correct = np.random.rand(len(workers)) <= probability
        incorrect = np.random.randint(0, latent.num_option - 1, size=len(workers))
        incorrect = np.where(incorrect >= true_answers, incorrect + 1, incorrect)
        answers = np.where(correct, true_answers, incorrect)
        return ClassificationData(
            num_worker=latent.num_worker, num_task=latent.num_task,
            num_option=latent.num_option, dim=self.dim,
            triple=torch.stack([torch.from_numpy(values) for values in (workers, answers, tasks)]),
            task_y=truth, task_y_mask=torch.ones(latent.num_task, dtype=torch.bool),
        ).setup()
