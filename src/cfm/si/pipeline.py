from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from cfm.audit.pipeline import masked_data
from cfm.data.crowd_data import CrowdData

from .adaptation import (
    AdaptationConfig,
    MechanismPosterior,
    adapt_mechanism,
    crossfit_e_value,
)


@dataclass(frozen=True)
class CrowdSIPipelineConfig:
    alpha: float = 0.05
    num_predictive_samples: int = 32
    audit_task_fraction: float = 1.0
    context_fraction: float = 0.5
    min_context_workers: int = 1
    min_audit_workers: int = 2
    min_worker_context_edges: int = 1
    adapt_only_with_evidence: bool = True
    require_trained_model: bool = True
    adaptation_steps: int = 100
    adaptation_learning_rate: float = 5e-2
    adaptation_likelihood_temperature: float = 1.0
    adaptation_kl_weight: float = 1.0
    adaptation_num_elbo_samples: int = 4
    adaptation_gradient_clip: float = 10.0

    def adaptation_config(self) -> AdaptationConfig:
        return AdaptationConfig(
            steps=self.adaptation_steps,
            learning_rate=self.adaptation_learning_rate,
            likelihood_temperature=self.adaptation_likelihood_temperature,
            kl_weight=self.adaptation_kl_weight,
            num_elbo_samples=self.adaptation_num_elbo_samples,
            gradient_clip=self.adaptation_gradient_clip,
        )


def run_crowdsi(
    model: torch.nn.Module,
    data: CrowdData,
    *,
    config: CrowdSIPipelineConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run zero-shot aggregation, evidence testing, and safe latent adaptation.

    Three predictions are returned separately:

    * original CrowdFM backbone output;
    * CrowdSI truth head at the amortized mechanism (zero-shot);
    * CrowdSI truth head at the evidence-selected mechanism.
    """

    cfg = config or CrowdSIPipelineConfig()
    trained = getattr(model, "crowdsi_trained", None)
    if cfg.require_trained_model and trained is not None and not bool(trained):
        raise RuntimeError("CrowdSI heads are marked untrained; refusing deployment adaptation")

    evidence = crossfit_e_value(
        model,
        data,
        adaptation_config=cfg.adaptation_config(),
        alpha=cfg.alpha,
        num_predictive_samples=cfg.num_predictive_samples,
        audit_task_fraction=cfg.audit_task_fraction,
        context_fraction=cfg.context_fraction,
        min_context_workers=cfg.min_context_workers,
        min_audit_workers=cfg.min_audit_workers,
        min_worker_context_edges=cfg.min_worker_context_edges,
        seed=seed,
    )
    context_data = masked_data(data, evidence.split.context_edge_mask)
    use_adaptation = evidence.adapt or not cfg.adapt_only_with_evidence
    selected_posterior: MechanismPosterior = evidence.base_posterior
    if use_adaptation:
        audit_mask = evidence.split.audit_edge_mask
        indices = torch.nonzero(audit_mask, as_tuple=False).flatten()
        selected_posterior = adapt_mechanism(
            model,
            context_data,
            data.triple[0, indices].long(),
            data.triple[2, indices].long(),
            data.triple[1, indices].long(),
            evidence.base_posterior,
            config=cfg.adaptation_config(),
            seed=seed + 1009,
        )

    model.eval()
    with torch.no_grad():
        zero_shot_output = model(
            data,
            mechanism_latent=evidence.base_posterior.mean,
            sample_mechanism=False,
        )
        selected_output = (
            zero_shot_output
            if not use_adaptation
            else model(
                data,
                mechanism_latent=selected_posterior.mean,
                sample_mechanism=False,
            )
        )

    crowdfm_posterior = torch.softmax(
        zero_shot_output["hat_task_option_base"], dim=-1
    )
    zero_shot_posterior = torch.softmax(
        zero_shot_output["hat_task_option"], dim=-1
    )
    selected_task_posterior = torch.softmax(
        selected_output["hat_task_option"], dim=-1
    )
    result: dict[str, Any] = {
        "config": asdict(cfg),
        "seed": seed,
        "e_value": evidence.e_value,
        "log_e_value": evidence.log_e_value,
        "e_value_threshold": evidence.threshold,
        "adaptation_supported": evidence.adapt,
        "used_adaptation": use_adaptation,
        "task_posterior": selected_task_posterior,
        "zero_shot_task_posterior": zero_shot_posterior,
        "crowdfm_task_posterior": crowdfm_posterior,
        "task_prediction": selected_task_posterior.argmax(dim=-1),
        "zero_shot_task_prediction": zero_shot_posterior.argmax(dim=-1),
        "crowdfm_task_prediction": crowdfm_posterior.argmax(dim=-1),
        "mechanism_mean_base": evidence.base_posterior.mean,
        "mechanism_log_variance_base": evidence.base_posterior.log_variance,
        "mechanism_mean_selected": selected_posterior.mean,
        "mechanism_log_variance_selected": selected_posterior.log_variance,
        "evidence": evidence,
    }
    task_y = getattr(data, "task_y", None)
    if isinstance(task_y, torch.Tensor):
        task_y = task_y.to(selected_task_posterior.device, dtype=torch.long)
        valid = task_y >= 0
        if torch.any(valid):
            result["accuracy"] = float(
                (result["task_prediction"][valid] == task_y[valid]).float().mean().item()
            )
            result["zero_shot_accuracy"] = float(
                (result["zero_shot_task_prediction"][valid] == task_y[valid])
                .float()
                .mean()
                .item()
            )
            result["crowdfm_accuracy"] = float(
                (result["crowdfm_task_prediction"][valid] == task_y[valid])
                .float()
                .mean()
                .item()
            )
    return result
