import torch


class TensorData:
    def to(self, device):
        for name, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self
