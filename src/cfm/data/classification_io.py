import pandas as pd
import torch

from .classification import ClassificationData
from .io import read_observations


def load_dataset(path, dim):
    labels, aligned, workers, tasks, nw, nt = read_observations(path)
    options, vocabulary = pd.factorize(labels["answer"], sort=False)
    mask = aligned.notna()
    category_ids = {value: index for index, value in enumerate(vocabulary)}
    if not aligned[mask].isin(vocabulary).all():
        raise ValueError("truth contains a category outside the observed vocabulary")
    truth = [category_ids[value] if present else 0 for value, present in zip(aligned, mask)]
    return ClassificationData(
        num_worker=nw, num_task=nt, num_option=len(vocabulary), dim=dim,
        triple=torch.tensor([workers.tolist(), options.tolist(), tasks.tolist()], dtype=torch.long),
        task_y=torch.tensor(truth, dtype=torch.long),
        task_y_mask=torch.tensor(mask.to_numpy(), dtype=torch.bool),
    ).setup()
