import torch


class CrowdData:
    def __init__(self, **kwargs):
        self.device = "cpu"
        self.dim = kwargs.get("dim", None)

        self.num_worker = kwargs.get("num_worker", None)
        self.num_task = kwargs.get("num_task", None)
        self.num_option = kwargs.get("num_option", None)

        self.triple = kwargs.get("triple", None)

        self.task_x = kwargs.get("task_x", None)
        self.worker_x = kwargs.get("worker_x", None)
        self.option_x = kwargs.get("option_x", None)

    def __getattr__(self, name):
        return None

    def to(self, device):
        import torch

        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        self.device = device
        return self

    def setup(self):
        self.get_degree()
        self.get_mv()
        self.reset_parameters()

    def get_mv(self):
        if self.triple is not None:
            import torch

            task_ids = self.triple[2, :]
            option_ids = self.triple[1, :]
            self.mv = torch.zeros((self.num_task, self.num_option), device=self.triple.device)
            self.mv.index_put_(
                (task_ids, option_ids),
                torch.ones_like(task_ids, dtype=self.mv.dtype),
                accumulate=True,
            )

    def get_degree(self):
        if self.triple is not None:
            worker_ids = self.triple[0, :]
            task_ids = self.triple[2, :]
            self.worker_degree = torch.bincount(worker_ids, minlength=self.num_worker)
            self.task_degree = torch.bincount(task_ids, minlength=self.num_task)

    def reset_parameters(self):
        if self.dim is not None:
            import torch

            if self.num_worker is not None:
                self.worker_x = torch.ones((self.num_worker, self.dim))
            if self.num_task is not None:
                self.task_x = torch.ones((self.num_task, self.dim))
            if self.num_option is not None:
                self.option_x = torch.randn((self.num_option, self.dim))

    def __reduce__(self):
        state = self.__dict__.copy()
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.cpu()
        return (self.__class__, (), state)

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[name] = value


def get_batch(data_list):
    batched_data = CrowdData(dim=data_list[0].dim)

    device = data_list[0].device
    total_worker, total_task, total_option = 0, 0, 0
    triples, worker_x, task_x, option_x = [], [], [], []
    for idx, data in enumerate(data_list):
        triple = data.triple.clone()
        triple[0] += total_worker
        triple[1] += total_option
        triple[2] += total_task

        triples.append(triple)
        worker_x.append(data.worker_x)
        task_x.append(data.task_x)
        option_x.append(data.option_x)

        total_worker += data.num_worker
        total_task += data.num_task
        total_option += data.num_option

    batched_data.triple = torch.cat(triples, dim=1).to(device)
    batched_data.worker_x = torch.cat(worker_x, dim=0).to(device)
    batched_data.task_x = torch.cat(task_x, dim=0).to(device)
    batched_data.option_x = torch.cat(option_x, dim=0).to(device)

    batched_data.num_worker = total_worker
    batched_data.num_task = total_task
    batched_data.num_option = total_option

    offset_worker, offset_task, offset_option = 0, 0, 0
    mask_workers, mask_tasks, mask_options = [], [], []
    for idx, data in enumerate(data_list):

        worker_mask = torch.zeros(total_worker, dtype=torch.bool)
        worker_mask[offset_worker : offset_worker + data.num_worker] = True
        mask_workers.append(worker_mask)

        task_mask = torch.zeros(total_task, dtype=torch.bool)
        task_mask[offset_task : offset_task + data.num_task] = True
        mask_tasks.append(task_mask)

        option_mask = torch.zeros(total_option, dtype=torch.bool)
        option_mask[offset_option : offset_option + data.num_option] = True
        mask_options.append(option_mask)

        offset_worker += data.num_worker
        offset_task += data.num_task
        offset_option += data.num_option

    batched_data.mask_workers = mask_workers
    batched_data.mask_tasks = mask_tasks
    batched_data.mask_options = mask_options

    batched_data.get_degree()

    return batched_data
