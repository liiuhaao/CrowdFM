from __future__ import annotations

import torch
import torch.nn.functional as F


def joint_annotation_log_likelihood(
    task_logits: torch.Tensor,
    emission_logits: torch.Tensor,
    observed_answers: torch.Tensor,
    query_tasks: torch.Tensor,
    *,
    reduction: str = "sum",
) -> torch.Tensor:
    """Compute the joint held-out response likelihood task by task.

    For a task k with held-out responses H_k, the likelihood is

        sum_c q_k(c) prod_{e in H_k} P_e(A_e | Y_k=c).

    The shared summation over one latent truth per task is essential; multiplying
    marginal edge probabilities would incorrectly assume held-out workers are
    independent after integrating out the task truth.
    """

    if task_logits.ndim != 2:
        raise ValueError("task_logits must have shape [num_tasks, num_options]")
    if emission_logits.ndim != 3:
        raise ValueError(
            "emission_logits must have shape [num_edges, num_truth, num_report]"
        )
    if emission_logits.shape[1:] != (
        task_logits.shape[1],
        task_logits.shape[1],
    ):
        raise ValueError("truth/report dimensions must match the task option count")
    if observed_answers.shape != (emission_logits.shape[0],):
        raise ValueError("one observed answer is required for every emission tensor")
    if query_tasks.shape != observed_answers.shape:
        raise ValueError("query_tasks and observed_answers must have matching shapes")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be one of: none, sum, mean")

    device = task_logits.device
    observed_answers = observed_answers.to(device=device, dtype=torch.long)
    query_tasks = query_tasks.to(device=device, dtype=torch.long)
    log_task = F.log_softmax(task_logits, dim=-1)
    log_emission = F.log_softmax(emission_logits, dim=-1)
    answer_log_probability_by_truth = log_emission.gather(
        2,
        observed_answers[:, None, None].expand(-1, emission_logits.shape[1], 1),
    ).squeeze(2)

    task_values: list[torch.Tensor] = []
    for task in torch.unique(query_tasks, sorted=True).tolist():
        edge_indices = torch.nonzero(query_tasks == task, as_tuple=False).flatten()
        conditional = answer_log_probability_by_truth[edge_indices].sum(dim=0)
        task_values.append(torch.logsumexp(log_task[task] + conditional, dim=-1))
    if not task_values:
        raise ValueError("at least one queried task is required")
    values = torch.stack(task_values)
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    return values.sum()


def supervised_emission_loss(
    emission_logits: torch.Tensor,
    observed_answers: torch.Tensor,
    query_tasks: torch.Tensor,
    task_truth: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for the true conditional-emission row on synthetic worlds."""

    query_tasks = query_tasks.to(emission_logits.device, dtype=torch.long)
    observed_answers = observed_answers.to(emission_logits.device, dtype=torch.long)
    task_truth = task_truth.to(emission_logits.device, dtype=torch.long)
    edge_truth = task_truth[query_tasks]
    known = edge_truth >= 0
    if not torch.any(known):
        raise ValueError("supervised_emission_loss requires at least one known task truth")
    edge_indices = torch.nonzero(known, as_tuple=False).flatten()
    selected = emission_logits[edge_indices, edge_truth[known]]
    return F.cross_entropy(selected, observed_answers[known])


def diagonal_gaussian_kl(
    mean_q: torch.Tensor,
    log_variance_q: torch.Tensor,
    mean_p: torch.Tensor | None = None,
    log_variance_p: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(N_q || N_p) for diagonal Gaussian distributions."""

    if mean_p is None:
        mean_p = torch.zeros_like(mean_q)
    if log_variance_p is None:
        log_variance_p = torch.zeros_like(log_variance_q)
    variance_ratio = torch.exp(log_variance_q - log_variance_p)
    squared_mean = (mean_q - mean_p).square() * torch.exp(-log_variance_p)
    return 0.5 * (
        log_variance_p
        - log_variance_q
        + variance_ratio
        + squared_mean
        - 1.0
    ).sum()


def symmetric_gaussian_kl(
    mean_a: torch.Tensor,
    log_variance_a: torch.Tensor,
    mean_b: torch.Tensor,
    log_variance_b: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        diagonal_gaussian_kl(mean_a, log_variance_a, mean_b, log_variance_b)
        + diagonal_gaussian_kl(mean_b, log_variance_b, mean_a, log_variance_a)
    )
