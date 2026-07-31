from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .CFM import CFM


class GaussianMechanismEncoder(torch.nn.Module):
    """Infer a permutation-invariant dataset-level crowd mechanism posterior.

    In addition to pooled node/edge embeddings, the encoder explicitly summarizes
    normalized worker and task degree distributions.  This preserves assignment
    density and selection structure that can be washed out by normalized graph
    attention.
    """

    def __init__(self, dim: int, latent_dim: int):
        super().__init__()
        self.dim = int(dim)
        self.latent_dim = int(latent_dim)
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(3 * self.dim, 2 * self.dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * self.dim, self.dim),
        )
        self.degree_encoder = torch.nn.Sequential(
            torch.nn.Linear(8, self.dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.dim, self.dim),
        )
        self.posterior = torch.nn.Sequential(
            torch.nn.Linear(5 * self.dim, 2 * self.dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * self.dim, 2 * self.latent_dim),
        )

    @staticmethod
    def _degree_statistics(
        workers: torch.Tensor,
        tasks: torch.Tensor,
        num_worker: int,
        num_task: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        worker_degree = torch.bincount(workers, minlength=num_worker).to(dtype)
        task_degree = torch.bincount(tasks, minlength=num_task).to(dtype)
        worker_degree = worker_degree / max(1, num_task)
        task_degree = task_degree / max(1, num_worker)

        def summarize(value: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [
                    value.mean(),
                    value.std(unbiased=False),
                    value.min(),
                    value.max(),
                ]
            )

        return torch.cat([summarize(worker_degree), summarize(task_degree)], dim=0)

    def forward(
        self,
        z_worker: torch.Tensor,
        z_task: torch.Tensor,
        z_option: torch.Tensor,
        triple: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if triple.ndim != 2 or triple.shape[0] != 3 or triple.shape[1] == 0:
            raise ValueError("a nonempty annotation triple with shape [3, num_edges] is required")
        workers, answers, tasks = triple.long()
        edge_features = torch.cat(
            [z_worker[workers], z_task[tasks], z_option[answers]],
            dim=-1,
        )
        edge_summary = self.edge_encoder(edge_features).mean(dim=0)
        degree_statistics = self._degree_statistics(
            workers,
            tasks,
            z_worker.shape[0],
            z_task.shape[0],
            z_worker.dtype,
        ).to(z_worker.device)
        degree_summary = self.degree_encoder(degree_statistics)
        summary = torch.cat(
            [
                z_worker.mean(dim=0),
                z_task.mean(dim=0),
                z_option.mean(dim=0),
                edge_summary,
                degree_summary,
            ],
            dim=-1,
        )
        mean, log_variance = self.posterior(summary).chunk(2, dim=-1)
        return mean, log_variance.clamp(min=-8.0, max=4.0)


class CompositionalMechanismBasis(torch.nn.Module):
    """Map a latent mechanism to a convex mixture of learned primitives."""

    def __init__(self, latent_dim: int, num_primitives: int, dim: int):
        super().__init__()
        self.router = torch.nn.Linear(int(latent_dim), int(num_primitives))
        self.primitives = torch.nn.Parameter(
            torch.randn(int(num_primitives), int(dim)) / max(1, int(dim)) ** 0.5
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 1:
            raise ValueError("mechanism_latent must be one-dimensional")
        weights = torch.softmax(self.router(latent), dim=-1)
        context = weights @ self.primitives
        return weights, context


class ConditionalEmissionHead(torch.nn.Module):
    """Predict P(A_ik=a | Y_k=c, context, worker i, task k)."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.dim = int(dim)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(5 * self.dim, 2 * self.dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(2 * self.dim, 1),
        )

    def forward(
        self,
        z_worker: torch.Tensor,
        z_task: torch.Tensor,
        z_option: torch.Tensor,
        mechanism_context: torch.Tensor,
        query_workers: torch.Tensor,
        query_tasks: torch.Tensor,
    ) -> torch.Tensor:
        query_workers = query_workers.to(z_worker.device, dtype=torch.long)
        query_tasks = query_tasks.to(z_task.device, dtype=torch.long)
        if query_workers.ndim != 1 or query_workers.shape != query_tasks.shape:
            raise ValueError("query_workers and query_tasks must be matching vectors")

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
        mechanism = mechanism_context[None, None, None, :].expand(
            num_query, num_option, num_option, -1
        )
        features = torch.cat(
            [worker, task, truth_option, reported_option, mechanism],
            dim=-1,
        )
        return self.scorer(features).squeeze(-1)


class CrowdSIFM(torch.nn.Module):
    """Crowd foundation model with amortized crowd system identification.

    The model augments the original CrowdFM backbone with

    * a dataset-level Gaussian mechanism posterior;
    * a compositional mechanism basis;
    * mechanism-adapted worker, task, and truth heads;
    * an edge-conditioned annotation emission law;
    * an annotation-assignment propensity head.

    At deployment, network weights remain fixed. Test-time adaptation updates only
    the low-dimensional mechanism posterior.
    """

    def __init__(self, **kwargs: Any):
        super().__init__()
        dim = int(kwargs["dim"])
        dropout = float(kwargs.get("dropout", 0.0))
        latent_dim = int(kwargs.get("mechanism_latent_dim", dim))
        num_primitives = int(kwargs.get("num_mechanism_primitives", 8))

        self.dim = dim
        self.mechanism_latent_dim = latent_dim
        self.backbone = CFM(**kwargs)
        self.mechanism_encoder = GaussianMechanismEncoder(dim, latent_dim)
        self.mechanism_basis = CompositionalMechanismBasis(
            latent_dim,
            num_primitives,
            dim,
        )
        self.worker_adapter = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 2 * dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * dim, dim),
            torch.nn.LayerNorm(dim),
        )
        self.task_adapter = torch.nn.Sequential(
            torch.nn.Linear(2 * dim, 2 * dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * dim, dim),
            torch.nn.LayerNorm(dim),
        )
        self.truth_scorer = torch.nn.Sequential(
            torch.nn.Linear(3 * dim, 2 * dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(2 * dim, 1),
        )
        self.emission_head = ConditionalEmissionHead(dim, dropout)
        self.assignment_head = torch.nn.Sequential(
            torch.nn.Linear(3 * dim, 2 * dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(2 * dim, 1),
        )
        self.register_buffer(
            "_crowdsi_trained",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    @property
    def crowdsi_trained(self) -> bool:
        return bool(self._crowdsi_trained.item())

    def mark_crowdsi_trained(self, trained: bool = True) -> None:
        self._crowdsi_trained.fill_(bool(trained))

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not frozen)

    def load_crowdfm_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint must contain a model_state_dict mapping")
        return self.backbone.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def reparameterize(
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        *,
        sample: bool,
    ) -> torch.Tensor:
        if not sample:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def forward(
        self,
        data: Any,
        *,
        query_workers: torch.Tensor | None = None,
        query_tasks: torch.Tensor | None = None,
        mechanism_latent: torch.Tensor | None = None,
        sample_mechanism: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if (query_workers is None) != (query_tasks is None):
            raise ValueError("query_workers and query_tasks must be provided together")

        base = self.backbone(data)
        mechanism_mean, mechanism_log_variance = self.mechanism_encoder(
            base["z_w"],
            base["z_t"],
            base["z_o"],
            data.triple,
        )
        if mechanism_latent is None:
            sample = self.training if sample_mechanism is None else bool(sample_mechanism)
            mechanism_latent = self.reparameterize(
                mechanism_mean,
                mechanism_log_variance,
                sample=sample,
            )
        mechanism_latent = mechanism_latent.to(
            device=base["z_w"].device,
            dtype=base["z_w"].dtype,
        )
        mechanism_weights, mechanism_context = self.mechanism_basis(mechanism_latent)

        worker_context = mechanism_context.expand(base["z_w"].shape[0], -1)
        task_context = mechanism_context.expand(base["z_t"].shape[0], -1)
        z_worker = self.worker_adapter(torch.cat([base["z_w"], worker_context], dim=-1))
        z_task = self.task_adapter(torch.cat([base["z_t"], task_context], dim=-1))

        num_task = z_task.shape[0]
        num_option = base["z_o"].shape[0]
        task = z_task[:, None, :].expand(-1, num_option, -1)
        option = base["z_o"][None, :, :].expand(num_task, -1, -1)
        mechanism = mechanism_context[None, None, :].expand(
            num_task, num_option, -1
        )
        task_logits = self.truth_scorer(
            torch.cat([task, option, mechanism], dim=-1)
        ).squeeze(-1)

        output = dict(base)
        output["hat_task_option_base"] = base["hat_task_option"]
        output["hat_task_option"] = task_logits
        output["z_worker_si"] = z_worker
        output["z_task_si"] = z_task
        output["mechanism_mean"] = mechanism_mean
        output["mechanism_log_variance"] = mechanism_log_variance
        output["mechanism_latent"] = mechanism_latent
        output["mechanism_weights"] = mechanism_weights
        output["mechanism_context"] = mechanism_context

        if query_workers is not None and query_tasks is not None:
            query_workers = query_workers.to(z_worker.device, dtype=torch.long)
            query_tasks = query_tasks.to(z_task.device, dtype=torch.long)
            output["hat_annotation_given_truth"] = self.emission_head(
                z_worker,
                z_task,
                base["z_o"],
                mechanism_context,
                query_workers,
                query_tasks,
            )
            assignment_features = torch.cat(
                [
                    z_worker[query_workers],
                    z_task[query_tasks],
                    mechanism_context.expand(query_workers.numel(), -1),
                ],
                dim=-1,
            )
            output["hat_assignment_logit"] = self.assignment_head(
                assignment_features
            ).squeeze(-1)
        return output
