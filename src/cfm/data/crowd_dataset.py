from torch.utils.data import IterableDataset

from cfm.data.crowd_simulator import CrowdSimulator


class CrowdDataset(IterableDataset):
    def __init__(self, **kwargs):
        self.simulator = CrowdSimulator(**kwargs)

    def __iter__(self):
        while True:
            yield self.simulator.generate()
