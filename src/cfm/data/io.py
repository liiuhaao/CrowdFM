from pathlib import Path

import pandas as pd


def read_observations(path):
    path = Path(path)
    labels = pd.read_csv(path / "label.csv")
    truths = pd.read_csv(path / "truth.csv")
    if not {"worker", "task", "answer"}.issubset(labels.columns):
        raise ValueError(f"missing label columns in {path}")
    if not {"task", "truth"}.issubset(truths.columns):
        raise ValueError(f"missing truth columns in {path}")
    if labels.empty or labels[["worker", "task", "answer"]].isna().any().any():
        raise ValueError("labels must be nonempty and have no missing observations")
    if truths["task"].isna().any() or truths["task"].duplicated().any():
        raise ValueError("truth task IDs must be present and unique")
    workers, worker_values = pd.factorize(labels["worker"], sort=False)
    tasks, task_values = pd.factorize(labels["task"], sort=False)
    truth_by_task = truths.set_index("task")["truth"]
    if not truth_by_task.index.isin(task_values).all():
        raise ValueError("truth contains tasks with no observations")
    return labels, truth_by_task.reindex(task_values), workers, tasks, len(worker_values), len(task_values)
