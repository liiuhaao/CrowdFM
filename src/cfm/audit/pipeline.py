from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from cfm.data.crowd_data import CrowdData

from .bootstrap import conditional_monte_carlo_test
from .disagreement import build_residual_matrix, estimate_confusion_matrices
from .split import CrossFitSplit, make_crossfit_split


@dataclass(frozen=True)
class CBRConfig:
    audit_fraction: float = 0.5
    context_fraction: float = 0.5
    min_context_workers: int = 1
    min_audit_workers: int = 2
    prior_strength: float = 10.0
    confusion_prior_mode: str = "global"
    min_pair_count: int = 1
    probability_clip: float = 1e-4
    variance_ridge: float = 1e-8
    num_bootstrap: int = 199
    alpha: float = 0.05
    refit_confusion_bootstrap: bool = False
    posterior_predictive_confusion_bootstrap: bool = True


def masked_data(data: CrowdData, edge_mask: torch.Tensor) -> CrowdData:
    """Create an edge-masked view while preserving the original node features."""

    masked = CrowdData(
        dim=data.dim,
        num_worker=data.num_worker,
        num_task=data.num_task,
        num_option=data.num_option,
        triple=data.triple[:, edge_mask].clone(),
    )
    if data.task_y is not None:
        masked.task_y = data.task_y.clone()
    masked.setup()
    for name in ("worker_x", "task_x", "option_x"):
        value = getattr(data, name, None)
        if isinstance(value, torch.Tensor):
            setattr(masked, name, value.clone())
    return masked.to(data.device)


# Backward-compatible private alias for early CbR code.
_masked_data = masked_data


def run_cbr_audit(
    model: torch.nn.Module,
    data: CrowdData,
    *,
    config: CBRConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one cross-fitted CbR audit on a CrowdFM dataset.

    The base model receives nuisance edges and context edges only. Confusion
    matrices are estimated on nuisance items. Residuals are constructed solely
    from held-out audit edges, preventing direct label leakage into q_k.

    The default estimator shrinks sparse worker/class confusion rows toward a
    leave-one-worker-out global class-conditional confusion matrix. The default
    posterior-predictive calibration propagates the remaining row uncertainty.
    """

    cfg = config or CBRConfig()
    if not 0.0 < cfg.alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    split: CrossFitSplit = make_crossfit_split(
        data.triple,
        data.num_task,
        audit_fraction=cfg.audit_fraction,
        context_fraction=cfg.context_fraction,
        min_context_workers=cfg.min_context_workers,
        min_audit_workers=cfg.min_audit_workers,
        seed=seed,
    )
    model_data = masked_data(data, split.model_input_edge_mask)

    model.eval()
    with torch.no_grad():
        output = model(model_data)
        task_posterior = torch.softmax(output["hat_task_option"], dim=-1)

    confusion = estimate_confusion_matrices(
        data.triple,
        task_posterior,
        num_worker=data.num_worker,
        num_option=data.num_option,
        edge_mask=split.nuisance_edge_mask,
        prior_strength=cfg.prior_strength,
        prior_mode=cfg.confusion_prior_mode,
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
        refit_confusion=cfg.refit_confusion_bootstrap,
        posterior_predictive_confusion=cfg.posterior_predictive_confusion_bootstrap,
        prior_strength=cfg.prior_strength,
        prior_mode=cfg.confusion_prior_mode,
        observed_statistic=residual.statistic,
        num_bootstrap=cfg.num_bootstrap,
        seed=seed,
        min_pair_count=cfg.min_pair_count,
        probability_clip=cfg.probability_clip,
        variance_ridge=cfg.variance_ridge,
    )

    return {
        "config": asdict(cfg),
        "seed": seed,
        "statistic": residual.statistic,
        "p_value": calibration.p_value,
        "reject": calibration.p_value <= cfg.alpha,
        "num_supported_pairs": int((residual.pair_counts >= cfg.min_pair_count).sum().item() // 2),
        "num_audit_edges": int(split.audit_edge_mask.sum().item()),
        "num_context_edges": int(split.context_edge_mask.sum().item()),
        "num_nuisance_edges": int(split.nuisance_edge_mask.sum().item()),
        "task_posterior": task_posterior,
        "confusion": confusion,
        "residual": residual,
        "split": split,
        "bootstrap": calibration,
    }
