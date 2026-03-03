import math

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax


class CFMEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CFMEncoder, self).__init__()
        self.dim = kwargs["dim"]
        self.layer = kwargs["layer"]
        self.head = kwargs["head"]
        self.dropout = kwargs["dropout"]
        self.device = kwargs["device"]

        self.x_worker = torch.nn.Parameter(torch.randn(self.dim, device=self.device), requires_grad=True)
        self.x_task = torch.nn.Parameter(torch.randn(self.dim, device=self.device), requires_grad=True)

        self.q_worker = torch.nn.ModuleList()
        self.k_worker = torch.nn.ModuleList()
        self.v_worker = torch.nn.ModuleList()
        self.q_task = torch.nn.ModuleList()
        self.k_task = torch.nn.ModuleList()
        self.v_task = torch.nn.ModuleList()

        self.out_worker = torch.nn.ModuleList()
        self.out_task = torch.nn.ModuleList()

        self.layer_norm = torch.nn.LayerNorm(self.dim)

        for _ in range(self.layer):
            self.q_worker.append(torch.nn.Linear(3 * self.dim, self.head * self.dim))
            self.k_worker.append(torch.nn.Linear(3 * self.dim, self.head * self.dim))
            self.v_worker.append(torch.nn.Linear(3 * self.dim, self.head * self.dim))
            self.out_worker.append(torch.nn.Linear(self.head * self.dim, self.dim))

            self.q_task.append(torch.nn.Linear(3 * self.dim, self.head * self.dim))
            self.k_task.append(torch.nn.Linear(3 * self.dim, self.head * self.dim))
            self.v_task.append(torch.nn.Linear(3 * self.dim, self.head * self.dim))
            self.out_task.append(torch.nn.Linear(self.head * self.dim, self.dim))

    def reset_parameters(self):
        def reset_module(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            elif isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
                for m in module:
                    reset_module(m)

        for _, module in self.__dict__.items():
            reset_module(module)

    def forward(self, data):
        worker_ids = data.triple[0]
        option_ids = data.triple[1]
        task_ids = data.triple[2]

        z_w = self.x_worker.expand(data.num_worker, -1)
        z_t = self.x_task.expand(data.num_task, -1)
        z_o = data.option_x

        for ly in range(self.layer):
            z = torch.cat([z_w[worker_ids], z_t[task_ids], z_o[option_ids]], dim=-1)
            # print(z.shape, self.head, self.dim, data.triple.shape)
            q_w = self.q_worker[ly](z).view(-1, self.head, self.dim)
            k_w = self.k_worker[ly](z).view(-1, self.head, self.dim)
            v_w = self.v_worker[ly](z).view(-1, self.head, self.dim)
            attn_score_w = (q_w * k_w).sum(-1) / math.sqrt(self.dim)  # [N, head]
            # deg_w = torch.log(data.worker_degree[worker_ids].unsqueeze(-1) + 1e-6)  # [N, 1]
            # attn_w = torch.sigmoid(attn_score_w - deg_w)  # [N, head]
            attn_w = softmax(attn_score_w, index=worker_ids, dim=0)
            # attn_w = softmax(torch.ones_like(attn_w), index=task_ids, dim=0)
            msg_w = attn_w.unsqueeze(-1) * v_w  # [N, head, dim]
            msg_w = msg_w.reshape(-1, self.head * self.dim)
            # msg_w = F.dropout(msg_w, p=self.dropout, training=self.training)
            msg_w = self.out_worker[ly](msg_w)

            q_t = self.q_task[ly](z).view(-1, self.head, self.dim)
            k_t = self.k_task[ly](z).view(-1, self.head, self.dim)
            v_t = self.v_task[ly](z).view(-1, self.head, self.dim)
            attn_score_t = (q_t * k_t).sum(-1) / math.sqrt(self.dim)  # [N, head]
            # deg_t = torch.log(data.task_degree[task_ids].unsqueeze(-1) + 1e-6)
            # attn_t = torch.sigmoid(attn_score_t - deg_t)  # [N, head]
            attn_t = softmax(attn_score_t, index=task_ids, dim=0)
            # attn_t = softmax(torch.ones_like(attn_t), index=worker_ids, dim=0)
            msg_t = attn_t.unsqueeze(-1) * v_t  # [N, head, dim]
            msg_t = msg_t.reshape(-1, self.head * self.dim)
            # msg_t = F.dropout(msg_t, p=self.dropout, training=self.training)
            msg_t = self.out_task[ly](msg_t)

            agg_w = scatter(msg_w, worker_ids, dim=0, reduce="sum", dim_size=data.num_worker)
            agg_t = scatter(msg_t, task_ids, dim=0, reduce="sum", dim_size=data.num_task)

            z_w = self.layer_norm(z_w + agg_w)
            z_t = self.layer_norm(z_t + agg_t)

        return z_w, z_t, z_o
