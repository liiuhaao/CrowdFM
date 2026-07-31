from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .bootstrap import conditional_monte_carlo_test
from .disagreement import (
    build_residual_matrix,
    estimate_confusion_matrices,
    item_conditioned_disagreement,
)
from .pipeline import CBRConfig, masked_data
from .split import CrossFitSplit, make_crossfit_split
from .synthetic import SyntheticWorld

CalibrationVariant = Literal[
    "oracle_q_oracle_p",
    "model_q_oracle_p",
    "oracle_q_estimated_p",
    "model_q_estimated_p",
]

CALIBRATION_VARIANTS: tuple[CalibrationVariant, ...] = (
    "oracle_q_oracle_p",
    "model_q_oracle_p",
    "oracle_q_estimated_p",
    "model_q_estimated_p",
)


@dataclass(frozen=True)
class CalibrationWorldResult:
    variant: CalibrationVariant
    statistic: float
    p_value: float
    reject: bool
    audit_posterior_accuracy: float
    confusion_mae: float
    disagreement_mae: float
    mean_nuisance_support: float
    median_nuisance_support: float
    min_nuisance_support: float
    num_supported_pairs: int
    num_audit_edges: int


def _oracle_posterior(world: SyntheticWorld, device: torch.device | str) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        world.truth.to(device),
        num_classes=world.config.num_option,
    ).float()


def _model_posterior(
    model: torch.nn.Module,
    world: SyntheticWorld,
    split: CrossFitSplit,
) -> torch.Tensor:
    model_data = masked_data(world.data, split.model_input_edge_mask)
    model.eval()
    with torch.no_grad():
        output = model(model_data)
    return torch.softmax(output["hat_task_option"], dim=-1)


def _worker_class_support(
    triple: torch.Tensor,
    task_posterior: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    num_worker: int,
) -> torch.Tensor:
    """Return effective nuisance observations for every worker-class row."""

    device = task_posterior.device
    triple = triple.to(device)
    edge_mask = edge_mask.to(device)
    support = torch.zeros(
        (num_worker, task_posterior.shape[1]),
        dtype=task_posterior.dtype,
        device=device,
    )
    workers = triple[0, edge_mask].long()
    tasks = triple[2, edge_mask].long()
    support.index_add_(0, workers, task_posterior[tasks])
    return support


def _audit_disagreement_mae(
    triple: torch.Tensor,
    task_posterior: torch.Tensor,
    estimated_confusion: torch.Tensor,
    true_confusion: torch.Tensor,
    audit_edge_mask: torch.Tensor,
) -> torch.Tensor:
    """Measure the error that directly enters the worker-pair residual."""

    device = task_posterior.device
    triple = triple.to(device)
    audit_edge_mask = audit_edge_mask.to(device)
    errors: list[torch.Tensor] = []
    audit_tasks = torch.unique(triple[2, audit_edge_mask].long())
    for task in audit_tasks.tolist():
        edge_indices = torch.nonzero(
            audit_edge_mask & (triple[2].long() == task), as_tuple=False
        ).flatten()
        if edge_indices.numel() < 2:
            continue
        local_pairs = torch.triu_indices(
            edge_indices.numel(), edge_indices.numel(), offset=1, device=device
        )
        left_edges = edge_indices[local_pairs[0]]
        right_edges = edge_indices[local_pairs[1]]
        worker_i = triple[0, left_edges].long()
        worker_j = triple[0, right_edges].long()
        distinct = worker_i != worker_j
        if not torch.any(distinct):
            continue
        worker_i = worker_i[distinct]
        worker_j = worker_j[distinct]
        task_ids = torch.full_like(worker_i, task)
        estimated = item_conditioned_disagreement(
            task_posterior,
            estimated_confusion,
            worker_i,
            worker_j,
            task_ids,
        )
        oracle = item_conditioned_disagreement(
            task_posterior,
            true_confusion,
            worker_i,
            worker_j,
            task_ids,
        )
        errors.append((estimated - oracle).abs())

    if not errors:
        return torch.tensor(float("nan"), device=device)
    return torch.cat(errors).mean()


def run_calibration_world(
    model: torch.nn.Module,
    world: SyntheticWorld,
    *,
    variant: CalibrationVariant,
    config: CBRConfig | None = None,
    seed: int = 0,
) -> CalibrationWorldResult:
    if variant not in CALIBRATION_VARIANTS:
        raise ValueError(f"Unknown calibration variant: {variant}")
    cfg = config or CBRConfig()
    data = world.data
    split = make_crossfit_split(
        data.triple,
        data.num_task,
        audit_fraction=cfg.audit_fraction,
        context_fraction=cfg.context_fraction,
        min_context_workers=cfg.min_context_workers,
        min_audit_workers=cfg.min_audit_workers,
        seed=seed,
    )

    use_model_q = variant.startswith("model_q")
    task_posterior = (
        _model_posterior(model, world, split)
        if use_model_q
        else _oracle_posterior(world, data.device)
    )

    use_oracle_p = variant.endswith("oracle_p")
    true_confusion = world.true_confusion.to(data.device)
    confusion = (
        true_confusion
        if use_oracle_p
        else estimate_confusion_matrices(
            data.triple,
            task_posterior,
            num_worker=data.num_worker,
            num_option=data.num_option,
            edge_mask=split.nuisance_edge_mask,
            prior_strength=cfg.prior_strength,
            prior_mode=cfg.confusion_prior_mode,
        )
    )

    residual = build_residual_matrix(
        data.triple,
        task_posterior,
        confusion,
        audit_edge_mask=split.audit_edge_mask,
        min_pair_count=cfg.min_pair_count,
        probability_clip=cfg.probability_clip,
        variance_ridge=cfg.variance_ridge,
    )
    calibration = conditional_monte_carlo_test(
        data.triple,
        task_posterior,
        confusion,
        audit_edge_mask=split.audit_edge_mask,
        nuisance_edge_mask=split.nuisance_edge_mask,
        refit_confusion=(not use_oracle_p and cfg.refit_confusion_bootstrap),
        posterior_predictive_confusion=(
            not use_oracle_p and cfg.posterior_predictive_confusion_bootstrap
        ),
        prior_strength=cfg.prior_strength,
        prior_mode=cfg.confusion_prior_mode,
        observed_statistic=residual.statistic,
        num_bootstrap=cfg.num_bootstrap,
        seed=seed,
        min_pair_count=cfg.min_pair_count,
        probability_clip=cfg.probability_clip,
        variance_ridge=cfg.variance_ridge,
    )

    audit_tasks = split.audit_task_mask
    posterior_prediction = task_posterior.argmax(dim=-1)
    posterior_accuracy = (
        posterior_prediction[audit_tasks] == world.truth.to(data.device)[audit_tasks]
    ).float().mean()
    confusion_mae = (confusion - true_confusion).abs().mean()
    disagreement_mae = _audit_disagreement_mae(
        data.triple,
        task_posterior,
        confusion,
        true_confusion,
        split.audit_edge_mask,
    )
    nuisance_support = _worker_class_support(
        data.triple,
        task_posterior,
        split.nuisance_edge_mask,
        num_worker=data.num_worker,
    ).flatten()

    return CalibrationWorldResult(
        variant=variant,
        statistic=residual.statistic,
        p_value=calibration.p_value,
        reject=calibration.p_value <= cfg.alpha,
        audit_posterior_accuracy=float(posterior_accuracy.item()),
        confusion_mae=float(confusion_mae.item()),
        disagreement_mae=float(disagreement_mae.item()),
        mean_nuisance_support=float(nuisance_support.mean().item()),
        median_nuisance_support=float(nuisance_support.median().item()),
        min_nuisance_support=float(nuisance_support.min().item()),
        num_supported_pairs=int(
            (residual.pair_counts >= cfg.min_pair_count).sum().item() // 2
        ),
        num_audit_edges=int(split.audit_edge_mask.sum().item()),
    )
