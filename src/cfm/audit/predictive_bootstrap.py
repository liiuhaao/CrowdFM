from __future__ import annotations

from dataclasses import dataclass

import torch

from .predictive import (
    PredictiveResidualResult,
    build_predictive_residual,
    normalize_annotation_probabilities,
)


@dataclass(frozen=True)
class PredictiveMonteCarloResult:
    observed: PredictiveResidualResult
    bootstrap_marginal_statistics: torch.Tensor
    bootstrap_dependence_statistics: torch.Tensor
    marginal_p_value: float
    dependence_p_value: float
    p_value: float


def conditional_predictive_test(
    task_posterior: torch.Tensor,
    emission_probabilities: torch.Tensor,
    observed_answers: torch.Tensor,
    query_workers: torch.Tensor,
    query_tasks: torch.Tensor,
    *,
    num_worker: int,
    num_bootstrap: int = 199,
    seed: int = 0,
    min_pair_count: int = 1,
    probability_clip: float = 1e-4,
    variance_ridge: float = 1e-8,
) -> PredictiveMonteCarloResult:
    """Test a fixed latent-truth annotation predictive law.

    For each replicate, one shared latent truth is sampled per task from q_k.
    Every held-out worker response is then sampled independently from its
    edge-conditioned emission row given that shared truth.  This preserves the
    within-task dependence implied by task-truth uncertainty.
    """

    if num_bootstrap < 1:
        raise ValueError("num_bootstrap must be positive")
    task_posterior = normalize_annotation_probabilities(
        task_posterior, probability_clip=probability_clip
    )
    emission_probabilities = normalize_annotation_probabilities(
        emission_probabilities, probability_clip=probability_clip
    )
    query_tasks = query_tasks.to(task_posterior.device, dtype=torch.long)

    observed = build_predictive_residual(
        task_posterior,
        emission_probabilities,
        observed_answers,
        query_workers,
        query_tasks,
        num_worker=num_worker,
        min_pair_count=min_pair_count,
        probability_clip=0.0,
        variance_ridge=variance_ridge,
    )

    generator = torch.Generator(device=task_posterior.device.type)
    generator.manual_seed(seed)
    marginal_statistics = torch.empty(num_bootstrap, dtype=torch.float64)
    dependence_statistics = torch.empty(num_bootstrap, dtype=torch.float64)
    edge_index = torch.arange(
        emission_probabilities.shape[0], device=task_posterior.device
    )

    for index in range(num_bootstrap):
        latent_truth = torch.multinomial(
            task_posterior,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(1)
        response_rows = emission_probabilities[
            edge_index,
            latent_truth[query_tasks],
        ]
        simulated_answers = torch.multinomial(
            response_rows,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(1)
        replicate = build_predictive_residual(
            task_posterior,
            emission_probabilities,
            simulated_answers,
            query_workers,
            query_tasks,
            num_worker=num_worker,
            min_pair_count=min_pair_count,
            probability_clip=0.0,
            variance_ridge=variance_ridge,
        )
        marginal_statistics[index] = replicate.marginal_statistic
        dependence_statistics[index] = replicate.dependence_statistic

    marginal_p_value = (
        1.0
        + int(
            (
                marginal_statistics
                >= observed.marginal_statistic - 1e-12
            ).sum().item()
        )
    ) / (num_bootstrap + 1.0)
    dependence_p_value = (
        1.0
        + int(
            (
                dependence_statistics
                >= observed.dependence_statistic - 1e-12
            ).sum().item()
        )
    ) / (num_bootstrap + 1.0)
    joint_p_value = min(1.0, 2.0 * min(marginal_p_value, dependence_p_value))

    return PredictiveMonteCarloResult(
        observed=observed,
        bootstrap_marginal_statistics=marginal_statistics,
        bootstrap_dependence_statistics=dependence_statistics,
        marginal_p_value=float(marginal_p_value),
        dependence_p_value=float(dependence_p_value),
        p_value=float(joint_p_value),
    )
