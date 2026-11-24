from torch.utils.data import IterableDataset


class SyntheticDataset(IterableDataset):
    """An infinite stream of scenes from either task's simulator."""

    def __init__(self, simulator):
        self.simulator = simulator

    def __iter__(self):
        while True:
            yield self.simulator.generate()


def collate_scenes(scenes):
    return scenes
