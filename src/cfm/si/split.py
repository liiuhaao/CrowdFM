from __future__ import annotations

from dataclasses import dataclass

import torch

from cfm.audit.predictive_split import make_annotation_audit_split


@dataclass(frozen=True)
class TaskCrossFitSplit:
    """Context graph plus two held-out folds with disjoint task identities."""

    context_edge_mask: torch.Tensor
    fold_a_edge_mask: torch.Tensor
    fold_b_edge_mask: torch.Tensor

    @property
    def audit_edge_mask(self) -> torch.Tensor:
        return self.fold_a_edge_mask | self.fold_b_edge_mask

    @property
    def fold_a_indices(self) -> torch.Tensor:
        return torch.nonzero(self.fold_a_edge_mask, as_tuple=False).flatten()

    @property
    def fold_b_indices(self) -> torch.Tensor:
        return torch.nonzero(self.fold_b_edge_mask, as_tuple=False).flatten()


def make_task_crossfit_split(
    triple: torch.Tensor,
    num_task: int,
    *,
    num_worker: int,
    audit_task_fraction: float = 1.0,
    context_fraction: float = 0.5,
    min_context_workers: int = 1,
    min_audit_workers: int = 2,
    min_worker_context_edges: int = 1,
    seed: int = 0,
) -> TaskCrossFitSplit:
    """Create two evidence folds that do not share latent task truths.

    Splitting individual edges from the same task across folds would leave the folds
    dependent through the unknown shared Y_k.  The e-value construction therefore
    assigns every audit edge of a task to one and only one fold.
    """

    base = make_annotation_audit_split(
        triple,
        num_task,
        num_worker=num_worker,
        audit_task_fraction=audit_task_fraction,
        context_fraction=context_fraction,
        min_context_workers=min_context_workers,
        min_audit_workers=min_audit_workers,
        min_worker_context_edges=min_worker_context_edges,
        seed=seed,
    )
    tasks = triple[2].long()
    audit_tasks = torch.unique(tasks[base.audit_edge_mask])
    if audit_tasks.numel() < 2:
        raise ValueError("CrowdSI cross-fitting requires at least two auditable tasks")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 104729)
    audit_tasks_cpu = audit_tasks.detach().cpu()
    permutation = audit_tasks_cpu[
        torch.randperm(audit_tasks_cpu.numel(), generator=generator)
    ]
    split_point = max(1, permutation.numel() // 2)
    split_point = min(split_point, permutation.numel() - 1)
    fold_a_tasks = permutation[:split_point].to(tasks.device)
    fold_b_tasks = permutation[split_point:].to(tasks.device)

    fold_a = base.audit_edge_mask & torch.isin(tasks, fold_a_tasks)
    fold_b = base.audit_edge_mask & torch.isin(tasks, fold_b_tasks)
    if not torch.any(fold_a) or not torch.any(fold_b):
        raise RuntimeError("both CrowdSI cross-fit folds must contain annotations")
    if torch.any(fold_a & fold_b):
        raise RuntimeError("cross-fit evidence folds overlap")
    if torch.any(base.context_edge_mask & (fold_a | fold_b)):
        raise RuntimeError("context and evidence folds overlap")

    task_overlap = torch.isin(
        torch.unique(tasks[fold_a]),
        torch.unique(tasks[fold_b]),
    )
    if torch.any(task_overlap):
        raise RuntimeError("cross-fit evidence folds share task identities")
    return TaskCrossFitSplit(
        context_edge_mask=base.context_edge_mask,
        fold_a_edge_mask=fold_a,
        fold_b_edge_mask=fold_b,
    )
