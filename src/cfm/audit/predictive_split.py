from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AnnotationAuditSplit:
    """Leakage-free context/audit edge partition for response prediction."""

    context_edge_mask: torch.Tensor
    audit_edge_mask: torch.Tensor
    audit_task_mask: torch.Tensor

    @property
    def model_input_edge_mask(self) -> torch.Tensor:
        return self.context_edge_mask

    @property
    def audit_edge_indices(self) -> torch.Tensor:
        return torch.nonzero(self.audit_edge_mask, as_tuple=False).flatten()


def make_annotation_audit_split(
    triple: torch.Tensor,
    num_task: int,
    *,
    num_worker: int | None = None,
    audit_task_fraction: float = 1.0,
    context_fraction: float = 0.5,
    min_context_workers: int = 1,
    min_audit_workers: int = 2,
    min_worker_context_edges: int = 1,
    seed: int = 0,
) -> AnnotationAuditSplit:
    """Split observed annotations into a context graph and held-out responses.

    Eligible tasks retain at least ``min_context_workers`` visible annotations
    and at least ``min_audit_workers`` held-out annotations.  The routine also
    guarantees that every worker remaining in the audit set has at least
    ``min_worker_context_edges`` visible annotations elsewhere in the context
    graph.  Tasks that cease to satisfy the audit-degree constraint are moved
    entirely back to context.
    """

    if triple.ndim != 2 or triple.shape[0] != 3:
        raise ValueError("triple must have shape [3, num_edges]")
    if num_task < 1:
        raise ValueError("num_task must be positive")
    if not 0.0 < audit_task_fraction <= 1.0:
        raise ValueError("audit_task_fraction must lie in (0, 1]")
    if not 0.0 < context_fraction < 1.0:
        raise ValueError("context_fraction must lie strictly between zero and one")
    if min_context_workers < 1:
        raise ValueError("min_context_workers must be positive")
    if min_audit_workers < 2:
        raise ValueError("min_audit_workers must be at least two")
    if min_worker_context_edges < 0:
        raise ValueError("min_worker_context_edges must be nonnegative")

    device = triple.device
    workers = triple[0].long()
    tasks = triple[2].long()
    if workers.numel() == 0:
        raise ValueError("at least one annotation edge is required")
    if num_worker is None:
        num_worker = int(workers.max().item()) + 1

    degree = torch.bincount(tasks, minlength=num_task)
    minimum_degree = min_context_workers + min_audit_workers
    eligible = torch.nonzero(degree >= minimum_degree, as_tuple=False).flatten()
    if eligible.numel() == 0:
        raise ValueError("no task has enough annotations for a context/audit split")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    eligible_cpu = eligible.detach().cpu()
    permutation = eligible_cpu[torch.randperm(eligible_cpu.numel(), generator=generator)]
    num_audit_tasks = max(1, int(round(audit_task_fraction * eligible_cpu.numel())))
    selected_tasks = permutation[:num_audit_tasks].to(device)

    context_mask = torch.ones(triple.shape[1], dtype=torch.bool, device=device)
    audit_mask = torch.zeros_like(context_mask)

    for task in selected_tasks.tolist():
        edge_indices = torch.nonzero(tasks == task, as_tuple=False).flatten()
        local_cpu = edge_indices.detach().cpu()
        local_permutation = local_cpu[
            torch.randperm(local_cpu.numel(), generator=generator)
        ].to(device)
        max_context = edge_indices.numel() - min_audit_workers
        proposed_context = int(round(context_fraction * edge_indices.numel()))
        num_context = max(min_context_workers, min(proposed_context, max_context))
        held_out = local_permutation[num_context:]
        context_mask[held_out] = False
        audit_mask[held_out] = True

    # A worker-specific response prediction is not meaningful when the worker has
    # no visible history.  Move enough of that worker's audit edges back to the
    # context graph to satisfy the requested global support floor.
    if min_worker_context_edges > 0:
        context_degree = torch.bincount(
            workers[context_mask], minlength=num_worker
        )
        for worker in range(num_worker):
            deficit = min_worker_context_edges - int(context_degree[worker].item())
            if deficit <= 0:
                continue
            candidates = torch.nonzero(
                audit_mask & (workers == worker), as_tuple=False
            ).flatten()
            if candidates.numel() == 0:
                continue
            moved = candidates[:deficit]
            context_mask[moved] = True
            audit_mask[moved] = False
            context_degree[worker] += moved.numel()

    # Preserve the minimum held-out degree after the worker-support repair.
    for task in selected_tasks.tolist():
        task_audit = audit_mask & (tasks == task)
        if int(task_audit.sum().item()) < min_audit_workers:
            context_mask[task_audit] = True
            audit_mask[task_audit] = False

    if not torch.any(audit_mask):
        raise ValueError("the support constraints leave no auditable annotations")
    if torch.any(context_mask & audit_mask):
        raise RuntimeError("context and audit masks overlap")
    if not torch.all(context_mask | audit_mask):
        raise RuntimeError("every annotation edge must belong to context or audit")

    actual_audit_tasks = torch.unique(tasks[audit_mask])
    audit_task_mask = torch.zeros(num_task, dtype=torch.bool, device=device)
    audit_task_mask[actual_audit_tasks] = True

    final_context_degree = torch.bincount(
        workers[context_mask], minlength=num_worker
    )
    audited_workers = torch.unique(workers[audit_mask])
    if audited_workers.numel() and torch.any(
        final_context_degree[audited_workers] < min_worker_context_edges
    ):
        raise RuntimeError("an audited worker lacks the required context support")

    return AnnotationAuditSplit(
        context_edge_mask=context_mask,
        audit_edge_mask=audit_mask,
        audit_task_mask=audit_task_mask,
    )
