from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CrossFitSplit:
    """Masks defining one leakage-free CbR audit split.

    Nuisance edges estimate annotator confusion matrices. Context edges on audit
    items are visible to the base aggregator. Audit edges are held out from the
    aggregator and are used only to construct the residual statistic.
    """

    nuisance_task_mask: torch.Tensor
    audit_task_mask: torch.Tensor
    nuisance_edge_mask: torch.Tensor
    context_edge_mask: torch.Tensor
    audit_edge_mask: torch.Tensor

    @property
    def model_input_edge_mask(self) -> torch.Tensor:
        return self.nuisance_edge_mask | self.context_edge_mask


def make_crossfit_split(
    triple: torch.Tensor,
    num_task: int,
    *,
    audit_fraction: float = 0.5,
    context_fraction: float = 0.5,
    min_context_workers: int = 1,
    min_audit_workers: int = 2,
    seed: int = 0,
) -> CrossFitSplit:
    """Create an item/edge cross-fit split without leaking audit labels.

    Tasks with too few labels to provide both context and at least two held-out
    audit workers are assigned to the nuisance set. Eligible tasks are randomly
    divided into nuisance and audit tasks. Each audit task is then split into
    visible context edges and held-out audit edges.
    """

    if triple.ndim != 2 or triple.shape[0] != 3:
        raise ValueError("triple must have shape [3, num_edges]")
    if not 0.0 < audit_fraction < 1.0:
        raise ValueError("audit_fraction must lie strictly between 0 and 1")
    if not 0.0 < context_fraction < 1.0:
        raise ValueError("context_fraction must lie strictly between 0 and 1")
    if min_context_workers < 1 or min_audit_workers < 2:
        raise ValueError("CbR requires >=1 context worker and >=2 audit workers")

    device = triple.device
    task_ids = triple[2].long()
    degree = torch.bincount(task_ids, minlength=num_task)
    min_degree = min_context_workers + min_audit_workers
    eligible = torch.nonzero(degree >= min_degree, as_tuple=False).flatten()

    if eligible.numel() < 2:
        raise ValueError(
            "At least two tasks with sufficient annotation degree are required "
            "to form nuisance and audit partitions."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    eligible_cpu = eligible.detach().cpu()
    permutation = eligible_cpu[torch.randperm(eligible_cpu.numel(), generator=generator)]

    num_audit = int(round(audit_fraction * eligible_cpu.numel()))
    num_audit = max(1, min(num_audit, eligible_cpu.numel() - 1))
    audit_tasks = permutation[:num_audit].to(device)

    audit_task_mask = torch.zeros(num_task, dtype=torch.bool, device=device)
    audit_task_mask[audit_tasks] = True
    nuisance_task_mask = ~audit_task_mask

    nuisance_edge_mask = nuisance_task_mask[task_ids]
    context_edge_mask = torch.zeros(triple.shape[1], dtype=torch.bool, device=device)
    audit_edge_mask = torch.zeros_like(context_edge_mask)

    for task in audit_tasks.tolist():
        edge_indices = torch.nonzero(task_ids == task, as_tuple=False).flatten()
        edge_indices_cpu = edge_indices.detach().cpu()
        local_perm = edge_indices_cpu[
            torch.randperm(edge_indices_cpu.numel(), generator=generator)
        ].to(device)

        max_context = edge_indices.numel() - min_audit_workers
        proposed_context = int(round(context_fraction * edge_indices.numel()))
        num_context = max(min_context_workers, min(proposed_context, max_context))

        context_edge_mask[local_perm[:num_context]] = True
        audit_edge_mask[local_perm[num_context:]] = True

    if torch.any(nuisance_edge_mask & context_edge_mask):
        raise RuntimeError("nuisance and context edge masks overlap")
    if torch.any(nuisance_edge_mask & audit_edge_mask):
        raise RuntimeError("nuisance and audit edge masks overlap")
    if torch.any(context_edge_mask & audit_edge_mask):
        raise RuntimeError("context and audit edge masks overlap")
    if not torch.all(nuisance_edge_mask | context_edge_mask | audit_edge_mask):
        raise RuntimeError("every annotation edge must belong to exactly one split")

    return CrossFitSplit(
        nuisance_task_mask=nuisance_task_mask,
        audit_task_mask=audit_task_mask,
        nuisance_edge_mask=nuisance_edge_mask,
        context_edge_mask=context_edge_mask,
        audit_edge_mask=audit_edge_mask,
    )
