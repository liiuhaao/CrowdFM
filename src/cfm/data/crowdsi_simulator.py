from __future__ import annotations

import math
import random

import torch
from torch.utils.data import IterableDataset

from cfm.data.crowd_data import CrowdData


class CrowdSISimulator:
    """Generate mechanism-diverse crowd worlds for system identification.

    The simulator deliberately varies both label noise and annotation assignment.
    It is not intended as a claim that these families exhaust real crowds; held-out
    families and compositions are required for evaluating mechanism generalization.
    """

    FAMILY_TO_ID = {
        "irt": 0,
        "class_bias": 1,
        "coalition": 2,
        "assignment_bias": 3,
        "mixed": 4,
    }

    def __init__(self, **kwargs):
        self.dim = int(kwargs.get("dim", 32))
        self.num_worker_range = tuple(kwargs.get("num_worker_range", (20, 100)))
        self.num_task_range = tuple(kwargs.get("num_task_range", (100, 300)))
        self.num_option_range = tuple(kwargs.get("num_option_range", (2, 20)))
        self.num_answer_each_task_range = tuple(
            kwargs.get("num_answer_each_task_range", (3, 10))
        )
        self.min_answers_per_task = int(kwargs.get("min_answers_per_task", 3))
        if self.min_answers_per_task < 1:
            raise ValueError("min_answers_per_task must be positive")
        self.mechanism_families = list(
            kwargs.get(
                "mechanism_families",
                ["irt", "class_bias", "coalition", "assignment_bias", "mixed"],
            )
        )
        unknown = set(self.mechanism_families) - set(self.FAMILY_TO_ID)
        if unknown:
            raise ValueError(f"unknown CrowdSI mechanism families: {sorted(unknown)}")
        self.D = 1.7

    @staticmethod
    def _standardize(value: torch.Tensor) -> torch.Tensor:
        return (value - value.mean()) / value.std(unbiased=False).clamp_min(1e-6)

    def _sample_dimensions(self, data: CrowdData) -> int:
        data.num_worker = random.randint(*self.num_worker_range)
        data.num_task = random.randint(*self.num_task_range)
        data.num_option = random.randint(*self.num_option_range)
        return random.randint(*self.num_answer_each_task_range)

    def generate(self) -> CrowdData:
        data = CrowdData(dim=self.dim)
        target_answers = self._sample_dimensions(data)
        family = random.choice(self.mechanism_families)

        ability_mu = random.uniform(-1.0, 1.0)
        ability_sigma = random.uniform(0.5, 2.0)
        difficulty_mu = random.uniform(-1.0, 1.0)
        difficulty_sigma = random.uniform(0.5, 2.0)
        class_bias_strength = (
            random.uniform(0.2, 0.8) if family in {"class_bias", "mixed"} else 0.0
        )
        coalition_fraction = (
            random.uniform(0.1, 0.5) if family in {"coalition", "mixed"} else 0.0
        )
        coalition_strength = (
            random.uniform(0.4, 0.95) if coalition_fraction > 0 else 0.0
        )
        assignment_strength = (
            random.uniform(0.5, 2.5)
            if family in {"assignment_bias", "mixed"}
            else 0.0
        )

        data.worker_ability = torch.randn(data.num_worker) * ability_sigma + ability_mu
        data.task_difficulty = (
            torch.randn(data.num_task) * difficulty_sigma + difficulty_mu
        )
        data.task_discrimination = torch.empty(data.num_task).uniform_(0.3, 3.0)
        guessing_base = 1.0 / data.num_option
        guessing_upper = max(guessing_base + 0.05, random.uniform(0.1, 0.4))
        data.task_guessing = torch.empty(data.num_task).uniform_(
            guessing_base,
            guessing_upper,
        )
        data.task_y = torch.randint(0, data.num_option, (data.num_task,))

        target_density = min(
            0.95,
            max(self.min_answers_per_task / data.num_worker, target_answers / data.num_worker),
        )
        base_logit = math.log(target_density / (1.0 - target_density))
        ability_z = self._standardize(data.worker_ability)[:, None]
        difficulty_z = self._standardize(data.task_difficulty)[None, :]
        assignment_logits = base_logit + assignment_strength * (ability_z - difficulty_z)
        assignment_probability = torch.sigmoid(assignment_logits)
        assignment_mask = torch.rand_like(assignment_probability) < assignment_probability
        minimum_answers = min(self.min_answers_per_task, data.num_worker)
        for task in range(data.num_task):
            if int(assignment_mask[:, task].sum().item()) < minimum_answers:
                top = torch.topk(
                    assignment_probability[:, task],
                    k=minimum_answers,
                ).indices
                assignment_mask[top, task] = True
        data.assignment_probability = assignment_probability
        data.assignment_mask = assignment_mask

        worker_ids, task_ids = torch.nonzero(
            assignment_mask,
            as_tuple=True,
        )
        truth = data.task_y[task_ids]
        theta = data.worker_ability[worker_ids]
        beta = data.task_difficulty[task_ids]
        discrimination = data.task_discrimination[task_ids]
        guessing = data.task_guessing[task_ids]
        probability_correct = guessing + (1.0 - guessing) * torch.sigmoid(
            self.D * discrimination * (theta - beta)
        )
        correct = torch.rand_like(probability_correct) <= probability_correct

        random_wrong = torch.randint(
            0,
            data.num_option - 1,
            (worker_ids.numel(),),
        )
        random_wrong = torch.where(random_wrong >= truth, random_wrong + 1, random_wrong)
        worker_group = torch.randint(0, 2, (data.num_worker,))
        offsets = 1 + worker_group[worker_ids].remainder(max(1, data.num_option - 1))
        preferred_wrong = (truth + offsets).remainder(data.num_option)
        use_preferred = torch.rand(worker_ids.numel()) < class_bias_strength
        wrong_answer = torch.where(use_preferred, preferred_wrong, random_wrong)
        answers = torch.where(correct, truth, wrong_answer)

        coalition_mask = torch.zeros(data.num_worker, dtype=torch.bool)
        if coalition_fraction > 0:
            coalition_size = max(1, int(round(coalition_fraction * data.num_worker)))
            coalition_workers = torch.randperm(data.num_worker)[:coalition_size]
            coalition_mask[coalition_workers] = True
            coalition_edges = coalition_mask[worker_ids]
            follow_coalition = coalition_edges & (
                torch.rand(worker_ids.numel()) < coalition_strength
            )
            coalition_answer = (truth + 1).remainder(data.num_option)
            answers = torch.where(follow_coalition, coalition_answer, answers)
        data.coalition_mask = coalition_mask
        data.worker_group = worker_group

        data.triple = torch.stack([worker_ids, answers, task_ids]).long()
        data.mechanism_family = self.FAMILY_TO_ID[family]
        data.mechanism_family_name = family
        data.mechanism_target = torch.tensor(
            [
                ability_mu,
                math.log(ability_sigma),
                difficulty_mu,
                math.log(difficulty_sigma),
                class_bias_strength,
                coalition_fraction,
                coalition_strength,
                assignment_strength,
            ],
            dtype=torch.float32,
        )
        data.setup()
        return data


class CrowdSIDataset(IterableDataset):
    def __init__(self, **kwargs):
        self.simulator = CrowdSISimulator(**kwargs)

    def __iter__(self):
        while True:
            yield self.simulator.generate()
