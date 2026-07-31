from __future__ import annotations

from dataclasses import dataclass

import torch

from .disagreement import (
    ConfusionPriorMode,
    build_residual_matrix,
    confusion_posterior_parameters,
    estimate_confusion_matrices,
)


@dataclass(frozen=True)
class MonteCarloResult:
    observed_statistic: float
    bootstrap_statistics: torch.Tensor
    p_value: float


def _sample_dirichlet_rows(
    concentration: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample independent Dirichlet rows with an explicit torch generator."""

    gamma = torch._standard_gamma(concentration, generator=generator)
    gamma = gamma.clamp_min(torch.finfo(concentration.dtype).tiny)
    return gamma / gamma.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(concentration.dtype).tiny
    )


def conditional_monte_carlo_test(
    triple: torch.Tensor,
    task_posterior: torch.Tensor,
    confusion: torch.Tensor,
    *,
    audit_edge_mask: torch.Tensor,
    nuisance_edge_mask: torch.Tensor | None = None,
    refit_confusion: bool = False,
    posterior_predictive_confusion: bool = False,
    prior_strength: float = 1.0,
    prior_mode: ConfusionPriorMode = "uniform",
    observed_statistic: float | None = None,
    num_bootstrap: int = 199,
    seed: int = 0,
    min_pair_count: int = 1,
    probability_clip: float = 1e-4,
    variance_ridge: float = 1e-8,
) -> MonteCarloResult:
    """Calibrate the residual statistic under a predictive null.

    Three confusion treatments are supported:

    - fixed: hold the supplied confusion matrix fixed;
    - refit: simulate nuisance labels and re-estimate confusion per replicate;
    - posterior predictive: draw confusion rows from their Dirichlet posterior,
      then compare observed and replicated audit discrepancies using the same
      draw.

    The item posterior remains fixed in all modes.
    """

    if num_bootstrap < 1:
        raise ValueError("num_bootstrap must be positive")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    if prior_mode not in ("uniform", "global"):
        raise ValueError("prior_mode must be 'uniform' or 'global'")
    if refit_confusion and posterior_predictive_confusion:
        raise ValueError("choose at most one confusion uncertainty mode")
    if (refit_confusion or posterior_predictive_confusion) and nuisance_edge_mask is None:
        raise ValueError(
            "nuisance_edge_mask is required when confusion uncertainty is enabled"
        )

    device = task_posterior.device
    triple = triple.to(device)
    audit_edge_mask = audit_edge_mask.to(device)
    if nuisance_edge_mask is not None:
        nuisance_edge_mask = nuisance_edge_mask.to(device)
        if nuisance_edge_mask.shape != audit_edge_mask.shape:
            raise ValueError("nuisance_edge_mask must have one entry per edge")
        if torch.any(nuisance_edge_mask & audit_edge_mask):
            raise ValueError("nuisance and audit edge masks must be disjoint")

    if observed_statistic is None:
        observed_statistic = build_residual_matrix(
            triple,
            task_posterior,
            confusion,
            audit_edge_mask=audit_edge_mask,
            min_pair_count=min_pair_count,
            probability_clip=probability_clip,
            variance_ridge=variance_ridge,
        ).statistic

    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    statistics = torch.empty(num_bootstrap, dtype=torch.float64)
    paired_exceedances = 0

    posterior_parameters = None
    if posterior_predictive_confusion:
        posterior_parameters = confusion_posterior_parameters(
            triple,
            task_posterior,
            num_worker=confusion.shape[0],
            num_option=confusion.shape[1],
            edge_mask=nuisance_edge_mask,
            prior_strength=prior_strength,
            prior_mode=prior_mode,
        )

    simulation_edge_mask = audit_edge_mask
    if refit_confusion:
        simulation_edge_mask = audit_edge_mask | nuisance_edge_mask

    simulation_workers = triple[0, simulation_edge_mask].long()
    simulation_tasks = triple[2, simulation_edge_mask].long()

    for index in range(num_bootstrap):
        replicate_confusion = confusion
        if posterior_predictive_confusion:
            replicate_confusion = _sample_dirichlet_rows(
                posterior_parameters,
                generator=generator,
            )

        latent_truth = torch.multinomial(
            task_posterior,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
        answer_probabilities = replicate_confusion[
            simulation_workers,
            latent_truth[simulation_tasks],
        ]
        simulated_answers = torch.multinomial(
            answer_probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)

        simulated_triple = triple.clone()
        simulated_triple[1, simulation_edge_mask] = simulated_answers

        if refit_confusion:
            replicate_confusion = estimate_confusion_matrices(
                simulated_triple,
                task_posterior,
                num_worker=confusion.shape[0],
                num_option=confusion.shape[1],
                edge_mask=nuisance_edge_mask,
                prior_strength=prior_strength,
                prior_mode=prior_mode,
            )

        replicate_statistic = build_residual_matrix(
            simulated_triple,
            task_posterior,
            replicate_confusion,
            audit_edge_mask=audit_edge_mask,
            min_pair_count=min_pair_count,
            probability_clip=probability_clip,
            variance_ridge=variance_ridge,
        ).statistic
        statistics[index] = replicate_statistic

        if posterior_predictive_confusion:
            observed_draw_statistic = build_residual_matrix(
                triple,
                task_posterior,
                replicate_confusion,
                audit_edge_mask=audit_edge_mask,
                min_pair_count=min_pair_count,
                probability_clip=probability_clip,
                variance_ridge=variance_ridge,
            ).statistic
            paired_exceedances += int(
                replicate_statistic >= observed_draw_statistic - 1e-12
            )

    if posterior_predictive_confusion:
        exceedances = paired_exceedances
    else:
        exceedances = int((statistics >= observed_statistic - 1e-12).sum().item())
    p_value = (1.0 + exceedances) / (num_bootstrap + 1.0)
    return MonteCarloResult(
        observed_statistic=float(observed_statistic),
        bootstrap_statistics=statistics,
        p_value=float(p_value),
    )
