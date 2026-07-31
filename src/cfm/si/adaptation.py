from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from cfm.audit.pipeline import masked_data
from cfm.data.crowd_data import CrowdData

from .likelihood import diagonal_gaussian_kl, joint_annotation_log_likelihood
from .split import TaskCrossFitSplit, make_task_crossfit_split


@dataclass(frozen=True)
class MechanismPosterior:
    mean: torch.Tensor
    log_variance: torch.Tensor
    objective_history: tuple[float, ...] = ()


@dataclass(frozen=True)
class AdaptationConfig:
    steps: int = 100
    learning_rate: float = 5e-2
    likelihood_temperature: float = 1.0
    kl_weight: float = 1.0
    num_elbo_samples: int = 4
    gradient_clip: float = 10.0


@dataclass(frozen=True)
class CrossFitEvidenceResult:
    split: TaskCrossFitSplit
    base_posterior: MechanismPosterior
    posterior_a: MechanismPosterior
    posterior_b: MechanismPosterior
    log_e_value_a_to_b: float
    log_e_value_b_to_a: float
    log_e_value: float
    e_value: float
    threshold: float
    adapt: bool


def _query_from_mask(
    data: CrowdData,
    edge_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.nonzero(edge_mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("an evidence fold contains no annotation edges")
    return (
        data.triple[0, indices].long(),
        data.triple[2, indices].long(),
        data.triple[1, indices].long(),
    )


def _sample_latents(
    posterior: MechanismPosterior,
    num_samples: int,
    *,
    seed: int,
) -> torch.Tensor:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    generator = torch.Generator(device=posterior.mean.device.type)
    generator.manual_seed(seed)
    noise = torch.randn(
        (num_samples, posterior.mean.numel()),
        generator=generator,
        device=posterior.mean.device,
        dtype=posterior.mean.dtype,
    )
    return posterior.mean.unsqueeze(0) + noise * torch.exp(
        0.5 * posterior.log_variance
    ).unsqueeze(0)


def _log_likelihood_at_latent(
    model: torch.nn.Module,
    context_data: CrowdData,
    query_workers: torch.Tensor,
    query_tasks: torch.Tensor,
    observed_answers: torch.Tensor,
    latent: torch.Tensor,
) -> torch.Tensor:
    output = model(
        context_data,
        query_workers=query_workers,
        query_tasks=query_tasks,
        mechanism_latent=latent,
        sample_mechanism=False,
    )
    return joint_annotation_log_likelihood(
        output["hat_task_option"],
        output["hat_annotation_given_truth"],
        observed_answers,
        query_tasks,
        reduction="sum",
    )


def adapt_mechanism(
    model: torch.nn.Module,
    context_data: CrowdData,
    query_workers: torch.Tensor,
    query_tasks: torch.Tensor,
    observed_answers: torch.Tensor,
    base_posterior: MechanismPosterior,
    *,
    config: AdaptationConfig | None = None,
    seed: int = 0,
) -> MechanismPosterior:
    """Update only a low-dimensional mechanism posterior; never update model weights."""

    cfg = config or AdaptationConfig()
    if cfg.steps < 1 or cfg.num_elbo_samples < 1:
        raise ValueError("adaptation steps and ELBO sample count must be positive")
    if cfg.learning_rate <= 0 or cfg.likelihood_temperature <= 0:
        raise ValueError("learning rate and likelihood temperature must be positive")
    if cfg.kl_weight < 0 or cfg.gradient_clip <= 0:
        raise ValueError("KL weight must be nonnegative and gradient clip positive")

    mean = torch.nn.Parameter(base_posterior.mean.detach().clone())
    log_variance = torch.nn.Parameter(base_posterior.log_variance.detach().clone())
    optimizer = torch.optim.Adam([mean, log_variance], lr=cfg.learning_rate)
    num_tasks = max(1, int(torch.unique(query_tasks).numel()))
    generator = torch.Generator(device=mean.device.type)
    generator.manual_seed(seed)

    parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
    was_training = model.training
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    history: list[float] = []
    try:
        for _ in range(cfg.steps):
            optimizer.zero_grad()
            noise = torch.randn(
                (cfg.num_elbo_samples, mean.numel()),
                generator=generator,
                device=mean.device,
                dtype=mean.dtype,
            )
            latents = mean.unsqueeze(0) + noise * torch.exp(
                0.5 * log_variance
            ).unsqueeze(0)
            likelihoods = torch.stack(
                [
                    _log_likelihood_at_latent(
                        model,
                        context_data,
                        query_workers,
                        query_tasks,
                        observed_answers,
                        latent,
                    )
                    for latent in latents
                ]
            )
            expected_log_likelihood = likelihoods.mean() / num_tasks
            kl = diagonal_gaussian_kl(
                mean,
                log_variance,
                base_posterior.mean,
                base_posterior.log_variance,
            ) / num_tasks
            objective = (
                expected_log_likelihood / cfg.likelihood_temperature
                - cfg.kl_weight * kl
            )
            (-objective).backward()
            torch.nn.utils.clip_grad_norm_([mean, log_variance], cfg.gradient_clip)
            optimizer.step()
            with torch.no_grad():
                log_variance.clamp_(min=-8.0, max=4.0)
            history.append(float(objective.detach().item()))
    finally:
        for parameter, flag in zip(model.parameters(), parameter_flags):
            parameter.requires_grad_(flag)
        model.train(was_training)

    return MechanismPosterior(
        mean=mean.detach(),
        log_variance=log_variance.detach(),
        objective_history=tuple(history),
    )


def posterior_predictive_log_likelihood(
    model: torch.nn.Module,
    context_data: CrowdData,
    query_workers: torch.Tensor,
    query_tasks: torch.Tensor,
    observed_answers: torch.Tensor,
    posterior: MechanismPosterior,
    *,
    num_samples: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """Log likelihood of a valid mixture predictive distribution."""

    latents = _sample_latents(posterior, num_samples, seed=seed)
    with torch.no_grad():
        values = torch.stack(
            [
                _log_likelihood_at_latent(
                    model,
                    context_data,
                    query_workers,
                    query_tasks,
                    observed_answers,
                    latent,
                )
                for latent in latents
            ]
        ).to(torch.float64)
    return torch.logsumexp(values, dim=0) - math.log(num_samples)


def point_predictive_log_likelihood(
    model: torch.nn.Module,
    context_data: CrowdData,
    query_workers: torch.Tensor,
    query_tasks: torch.Tensor,
    observed_answers: torch.Tensor,
    latent: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return _log_likelihood_at_latent(
            model,
            context_data,
            query_workers,
            query_tasks,
            observed_answers,
            latent,
        ).to(torch.float64)


def crossfit_e_value(
    model: torch.nn.Module,
    data: CrowdData,
    *,
    adaptation_config: AdaptationConfig | None = None,
    alpha: float = 0.05,
    num_predictive_samples: int = 32,
    audit_task_fraction: float = 1.0,
    context_fraction: float = 0.5,
    min_context_workers: int = 1,
    min_audit_workers: int = 2,
    min_worker_context_edges: int = 1,
    seed: int = 0,
) -> CrossFitEvidenceResult:
    """Test whether mechanism adaptation improves held-out predictive evidence.

    Each numerator is learned on one task-disjoint fold and evaluated on the
    other.  The denominator is the fixed plug-in law at the amortized mechanism
    mean.  Under that fixed null and independent tasks, each likelihood ratio is
    an e-value; their arithmetic mean is also an e-value.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    split = make_task_crossfit_split(
        data.triple,
        data.num_task,
        num_worker=data.num_worker,
        audit_task_fraction=audit_task_fraction,
        context_fraction=context_fraction,
        min_context_workers=min_context_workers,
        min_audit_workers=min_audit_workers,
        min_worker_context_edges=min_worker_context_edges,
        seed=seed,
    )
    context_data = masked_data(data, split.context_edge_mask)
    model.eval()
    with torch.no_grad():
        initial = model(context_data, sample_mechanism=False)
    base = MechanismPosterior(
        mean=initial["mechanism_mean"].detach(),
        log_variance=initial["mechanism_log_variance"].detach(),
    )

    workers_a, tasks_a, answers_a = _query_from_mask(data, split.fold_a_edge_mask)
    workers_b, tasks_b, answers_b = _query_from_mask(data, split.fold_b_edge_mask)
    posterior_a = adapt_mechanism(
        model,
        context_data,
        workers_a,
        tasks_a,
        answers_a,
        base,
        config=adaptation_config,
        seed=seed + 1,
    )
    posterior_b = adapt_mechanism(
        model,
        context_data,
        workers_b,
        tasks_b,
        answers_b,
        base,
        config=adaptation_config,
        seed=seed + 2,
    )

    numerator_b = posterior_predictive_log_likelihood(
        model,
        context_data,
        workers_b,
        tasks_b,
        answers_b,
        posterior_a,
        num_samples=num_predictive_samples,
        seed=seed + 3,
    )
    denominator_b = point_predictive_log_likelihood(
        model,
        context_data,
        workers_b,
        tasks_b,
        answers_b,
        base.mean,
    )
    numerator_a = posterior_predictive_log_likelihood(
        model,
        context_data,
        workers_a,
        tasks_a,
        answers_a,
        posterior_b,
        num_samples=num_predictive_samples,
        seed=seed + 4,
    )
    denominator_a = point_predictive_log_likelihood(
        model,
        context_data,
        workers_a,
        tasks_a,
        answers_a,
        base.mean,
    )

    log_e_a_to_b = numerator_b - denominator_b
    log_e_b_to_a = numerator_a - denominator_a
    log_e = torch.logsumexp(
        torch.stack([log_e_a_to_b, log_e_b_to_a]),
        dim=0,
    ) - math.log(2.0)
    e_value = float(torch.exp(log_e.clamp(max=700.0)).item())
    threshold = 1.0 / alpha
    return CrossFitEvidenceResult(
        split=split,
        base_posterior=base,
        posterior_a=posterior_a,
        posterior_b=posterior_b,
        log_e_value_a_to_b=float(log_e_a_to_b.item()),
        log_e_value_b_to_a=float(log_e_b_to_a.item()),
        log_e_value=float(log_e.item()),
        e_value=e_value,
        threshold=threshold,
        adapt=e_value >= threshold,
    )
