import random

import numpy as np
import torch

from cfm.data.crowd_data import CrowdData


class CrowdSimulator:
    def __init__(self, **kwargs):
        self.dim = kwargs.get("dim", 5)
        self.num_worker_range = kwargs.get("num_worker_range", (20, 100))
        self.num_task_range = kwargs.get("num_task_range", (100, 300))
        self.num_option_range = kwargs.get("num_option_range", (2, 20))
        self.num_answer_each_task_range = kwargs.get("num_answer_each_task_range", (2, 10))
        self.alpha_range = kwargs.get("alpha_range", (0, 30))
        self.D = 1.7

        # worker ability: θ_i ~ N(μ, σ²)
        self.ability_mu_range = kwargs.get("ability_mu_range", (-1.0, 1.0))
        self.ability_sigma_range = kwargs.get("ability_sigma_range", (0.5, 2.0))

        # task difficulty: β_j ~ N(μ, σ²)
        self.difficulty_mu_range = kwargs.get("difficulty_mu_range", (-1.0, 1.0))
        self.difficulty_sigma_range = kwargs.get("difficulty_sigma_range", (0.5, 2.0))

        # discrimination: α_j ~ Uniform(a, b)
        self.discrimination_min_range = kwargs.get("discrimination_min_range", (0.3, 1.0))
        self.discrimination_max_range = kwargs.get("discrimination_max_range", (1.5, 3.0))

        # guessing base: base = 1/K, then c_j ~ Uniform(base, upper), upper ∈ [0.1, 0.4]
        self.guessing_upper_range = kwargs.get("guessing_upper_range", (0.1, 0.4))

    def _simulate_num(self, data):
        data.num_worker = random.randint(self.num_worker_range[0], self.num_worker_range[1])
        data.num_task = random.randint(self.num_task_range[0], self.num_task_range[1])
        data.num_option = random.randint(self.num_option_range[0], self.num_option_range[1])
        data.num_answer_each_task = random.randint(self.num_answer_each_task_range[0], self.num_answer_each_task_range[1])

    def _simulate_worker(self, data):
        mu = random.uniform(*self.ability_mu_range)
        sigma = random.uniform(*self.ability_sigma_range)
        data.worker_ability = torch.randn(data.num_worker) * sigma + mu

    def _simulate_task(self, data):
        data.task_y = torch.randint(0, data.num_option, (data.num_task,))

        # task_difficulty
        mu_d = random.uniform(*self.difficulty_mu_range)
        sigma_d = random.uniform(*self.difficulty_sigma_range)
        data.task_difficulty = torch.randn(data.num_task) * sigma_d + mu_d

        # task_discrimination
        a = random.uniform(*self.discrimination_min_range)
        b = random.uniform(*self.discrimination_max_range)
        if a >= b:
            a, b = b, a  # ensure a < b
        data.task_discrimination = torch.rand(data.num_task) * (b - a) + a

        # guessing
        guessing_base = 1.0 / data.num_option
        guessing_upper = random.uniform(*self.guessing_upper_range)
        if guessing_base >= guessing_upper:
            guessing_upper = max(guessing_upper, guessing_base + 0.05)
        data.task_guessing = torch.rand(data.num_task) * (guessing_upper - guessing_base) + guessing_base

    def _simulate_crowd(self, data):
        # Sample number of answers per task
        num_answers = np.random.poisson(lam=data.num_answer_each_task, size=data.num_task)
        num_answers = np.clip(num_answers, a_min=1, a_max=data.num_worker)

        # Randomly permute worker indices for each task
        perms = np.random.rand(data.num_task, data.num_worker).argsort(axis=1)

        # Select worker ids for answers
        max_answer = num_answers.max()
        answer_mask = num_answers[:, None] > np.arange(max_answer)[None, :]
        workers_2d = perms[:, :max_answer]
        worker_ids = workers_2d[answer_mask]
        task_ids = np.repeat(np.arange(data.num_task), num_answers)

        # 3PL
        theta = data.worker_ability.numpy()[worker_ids]  # θ_i
        beta = data.task_difficulty.numpy()[task_ids]  # β_j
        alpha = data.task_discrimination.numpy()[task_ids]  # α_j
        c = data.task_guessing.numpy()[task_ids]  # c_j
        logit = self.D * alpha * (theta - beta)
        sigmoid = 1 / (1 + np.exp(-logit))
        prob_correct = c + (1 - c) * sigmoid

        correct_mask = np.random.rand(len(worker_ids)) <= prob_correct

        task_y = data.task_y.numpy()
        y_true = task_y[task_ids]

        incorrect_ans = np.random.randint(0, data.num_option - 1, size=len(worker_ids))
        incorrect_ans = np.where(incorrect_ans >= y_true, incorrect_ans + 1, incorrect_ans)

        option_ids = np.where(correct_mask, y_true, incorrect_ans)

        worker_ids, option_ids, task_ids = torch.from_numpy(worker_ids), torch.from_numpy(option_ids), torch.from_numpy(task_ids)
        data.triple = torch.stack([worker_ids, option_ids, task_ids])

    def generate(self):
        data = CrowdData(dim=self.dim)
        self._simulate_num(data)
        self._simulate_worker(data)
        self._simulate_task(data)
        self._simulate_crowd(data)
        data.setup()
        return data
