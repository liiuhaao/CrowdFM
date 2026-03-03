import torch
import torch.nn.functional as F

from cfm.data.crowd_data import get_batch

from .CFMEncoder import CFMEncoder


class CFM(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CFM, self).__init__()
        self.dim = kwargs["dim"]
        self.layer = kwargs["layer"]

        self.encoder = CFMEncoder(**kwargs)

        self.task_option_classifactor = torch.nn.Sequential(
            torch.nn.Linear(2 * self.dim, 2 * self.dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * self.dim, 1),
        )

    def reset_parameters(self):
        def reset_module(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            elif isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
                for m in module:
                    reset_module(m)

        for _, module in self.__dict__.items():
            reset_module(module)

    def batch_forward(self, data_list):
        batch_data = get_batch(data_list)
        batch_z_w, batch_z_t, batch_z_o = self.encoder(batch_data)
        out_list = []
        for idx, data in enumerate(data_list):
            z_w, z_t, z_o = (
                batch_z_w[batch_data.mask_workers[idx]],
                batch_z_t[batch_data.mask_tasks[idx]],
                batch_z_o[batch_data.mask_options[idx]],
            )
            out_list.append(self.forward(data, z_w=z_w, z_t=z_t, z_o=z_o))
        return out_list

    def forward(self, data, z_w=None, z_t=None, z_o=None):
        if z_w is None:
            z_w, z_t, z_o = self.encoder(data)

        z_task_option = torch.cat(
            [
                z_t.unsqueeze(1).expand(-1, data.num_option, self.dim),
                z_o.unsqueeze(0).expand(data.num_task, -1, self.dim),
            ],
            -1,
        )
        hat_task_option = self.task_option_classifactor(z_task_option).squeeze(-1)

        out = {
            "z_w": z_w,
            "z_t": z_t,
            "z_o": z_o,
            "hat_task_option": hat_task_option,
        }

        return out

    def batch_eval(self, data_list, size=None):
        out_list = self.batch_forward(data_list)
        res_list = []
        res_eval = []
        with torch.no_grad():
            for data, out in zip(data_list, out_list):
                eval_result = self.evaluate(data, out)
                res_list.append(data)
                res_eval.append(eval_result)
                if size is not None and len(res_list) >= size:
                    break
        return res_list, res_eval

    def batch_loss(self, data_list):
        out_list = self.batch_forward(data_list)
        losses = [sum(self.loss(data, out).values()) for data, out in zip(data_list, out_list)]
        return torch.stack(losses).mean()

    def loss(self, data, out=None):
        out = self.forward(data) if out is None else out
        valid_task_mask = data.task_y != -1

        loss_dict = dict()
        loss_dict["loss_task_option"] = F.cross_entropy(out["hat_task_option"][valid_task_mask], data.task_y[valid_task_mask])
        return loss_dict

    def get_mv_pred(self, data):
        max_mask = (data.mv == data.mv.max(dim=1, keepdim=True).values).float()
        y_pred = torch.multinomial(max_mask, num_samples=1).squeeze(1)
        return y_pred

    @torch.no_grad()
    def evaluate(self, data, out=None):
        out = self.forward(data) if out is None else out
        valid_task_mask = data.task_y != -1

        perform = dict()

        # loss
        perform.update({k: v.item() for k, v in self.loss(data, out).items()})

        # task_y_acc
        task_y_pred = torch.argmax(out["hat_task_option"], dim=1)
        perform["task_y_acc"] = (task_y_pred[valid_task_mask] == data.task_y[valid_task_mask]).float().mean().item()
        return perform
