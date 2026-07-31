from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .CFM import CFM


class ConditionalAnnotationHead(torch.nn.Module):
    """Predict a held-out response conditional on each candidate task truth.

    For every query edge (worker i, task k), the shared scorer evaluates every
    candidate truth c and reported option a.  The output has shape
    [num_queries, num_options, num_options] and is normalized over the final
    reported-option axis.  Sharing the scorer across option embeddings preserves
    the variable-K interface of CrowdFM.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(4 * self.dim, 2 * self.dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(2 * self.dim, 1),
        )

    def forward(
        self,
        z_worker: torch.Tensor,
        z_task: torch.Tensor,
        z_option: torch.Tensor,
        query_workers: torch.Tensor,
        query_tasks: torch.Tensor,
    ) -> torch.Tensor:
        if query_workers.ndim != 1 or query_tasks.ndim != 1:
            raise ValueError("query_workers and query_tasks must be one-dimensional")
        if query_workers.shape != query_tasks.shape:
            raise ValueError("query_workers and query_tasks must have the same shape")
        if z_worker.ndim != 2 or z_task.ndim != 2 or z_option.ndim != 2:
            raise ValueError("worker, task, and option embeddings must be matrices")
        if not (z_worker.shape[1] == z_task.shape[1] == z_option.shape[1] == self.dim):
            raise ValueError("all embedding widths must match the response-head dimension")

        query_workers = query_workers.to(device=z_worker.device, dtype=torch.long)
        query_tasks = query_tasks.to(device=z_task.device, dtype=torch.long)
        num_query = query_workers.numel()
        num_option = z_option.shape[0]

        worker = z_worker[query_workers, None, None, :].expand(
            -1, num_option, num_option, -1
        )
        task = z_task[query_tasks, None, None, :].expand(
            -1, num_option, num_option, -1
        )
        truth_option = z_option[None, :, None, :].expand(
            num_query, -1, num_option, -1
        )
        reported_option = z_option[None, None, :, :].expand(
            num_query, num_option, -1, -1
        )
        features = torch.cat(
            [worker, task, truth_option, reported_option],
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)


class PredictiveCFM(torch.nn.Module):
    """CrowdFM with a learned annotation-emission distribution.

    The original backbone predicts q_theta(Y_k | context graph).  The added head
    predicts

        P_phi(A_ik = a | Y_k = c, context graph, worker i, task k).

    Marginal response probabilities and within-task worker dependence are then
    induced by the shared task posterior.  The head must be trained with masked
    annotations before its probabilities are used by the deployment audit.
    """

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.backbone = CFM(**kwargs)
        self.annotation_head = ConditionalAnnotationHead(
            dim=int(kwargs["dim"]),
            dropout=float(kwargs.get("dropout", 0.0)),
        )
        self.register_buffer(
            "_response_head_trained",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    @property
    def response_head_trained(self) -> bool:
        return bool(self._response_head_trained.item())

    def mark_response_head_trained(self, trained: bool = True) -> None:
        self._response_head_trained.fill_(bool(trained))

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen)

    def load_crowdfm_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load an original CrowdFM checkpoint into the backbone only."""

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint must contain a model_state_dict mapping")
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def forward(
        self,
        data: Any,
        *,
        query_workers: torch.Tensor | None = None,
        query_tasks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        output = self.backbone(data)
        if (query_workers is None) != (query_tasks is None):
            raise ValueError("query_workers and query_tasks must be provided together")
        if query_workers is not None and query_tasks is not None:
            output["hat_annotation_given_truth"] = self.annotation_head(
                output["z_w"],
                output["z_t"],
                output["z_o"],
                query_workers,
                query_tasks,
            )
        return output

    @staticmethod
    def supervised_annotation_loss(
        emission_logits: torch.Tensor,
        true_task_labels: torch.Tensor,
        observed_answers: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy for synthetic edges whose latent truth is known."""

        if emission_logits.ndim != 3:
            raise ValueError(
                "emission_logits must have shape [num_edges, num_truth, num_report]"
            )
        true_task_labels = true_task_labels.to(
            device=emission_logits.device,
            dtype=torch.long,
        )
        observed_answers = observed_answers.to(
            device=emission_logits.device,
            dtype=torch.long,
        )
        if true_task_labels.shape != (emission_logits.shape[0],):
            raise ValueError("one true task label is required for every query edge")
        selected_rows = emission_logits[
            torch.arange(emission_logits.shape[0], device=emission_logits.device),
            true_task_labels,
        ]
        return F.cross_entropy(selected_rows, observed_answers)
