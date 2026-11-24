import torch

from .message_passing import aggregate_attention


class BipartiteEncoder(torch.nn.Module):
    """Worker/task embeddings conditioned on task-specific edge features."""

    def __init__(self, dim, layer, head, dropout, aggregation=aggregate_attention):
        super().__init__()
        if min(dim, layer, head) < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid encoder dimensions or dropout")
        self.dim, self.head, self.dropout = dim, head, dropout
        self.aggregation = aggregation
        self.worker_embedding = torch.nn.Parameter(torch.empty(dim))
        self.task_embedding = torch.nn.Parameter(torch.empty(dim))
        self.layers = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                f"{side}_{kind}": torch.nn.Linear(3 * dim, head * dim)
                for side in ("worker", "task") for kind in ("q", "k", "v")
            } | {
                f"{side}_out": torch.nn.Linear(head * dim, dim)
                for side in ("worker", "task")
            } | {
                f"{side}_norm": torch.nn.LayerNorm(dim)
                for side in ("worker", "task")
            }) for _ in range(layer)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.worker_embedding)
        torch.nn.init.normal_(self.task_embedding)
        for module in self.layers.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
                module.reset_parameters()

    def forward(self, edge_index, edge_features, num_worker, num_task):
        workers, tasks = edge_index
        zw = self.worker_embedding.expand(num_worker, -1)
        zt = self.task_embedding.expand(num_task, -1)
        for layer in self.layers:
            context = torch.cat((zw[workers], zt[tasks], edge_features), dim=-1)
            updates = {}
            for side, ids, count in (("worker", workers, num_worker), ("task", tasks, num_task)):
                updates[side] = self.aggregation(
                    context, ids, count,
                    layer[f"{side}_q"], layer[f"{side}_k"], layer[f"{side}_v"], layer[f"{side}_out"],
                    self.dim, self.head, self.dropout, self.training,
                )
            zw = layer["worker_norm"](zw + updates["worker"])
            zt = layer["task_norm"](zt + updates["task"])
        return zw, zt
