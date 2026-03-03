from pathlib import Path

import pandas as pd
import torch

from cfm.data.crowd_data import CrowdData


def load_single_data(cfg, data_path):
    data = CrowdData(dim=cfg.dim)
    label_df = pd.read_csv(data_path / "label.csv")
    truth_df = pd.read_csv(data_path / "truth.csv")

    w2wid, t2tid, o2oid = {}, {}, {}

    worker_ids, task_ids, option_ids = [], [], []
    for i, row in enumerate(label_df.itertuples()):
        worker, task, option = row.worker, row.task, row.answer
        if worker not in w2wid:
            w2wid[worker] = len(w2wid)
        if task not in t2tid:
            t2tid[task] = len(t2tid)
        if option not in o2oid:
            o2oid[option] = len(o2oid)
        worker_ids.append(w2wid[worker])
        task_ids.append(t2tid[task])
        option_ids.append(o2oid[option])
    data.triple = torch.tensor([worker_ids, option_ids, task_ids])

    task_y = [-1 for _ in range(len(t2tid))]
    for i, row in enumerate(truth_df.itertuples()):
        task, truth = row.task, row.truth
        if truth not in o2oid:
            o2oid[truth] = len(o2oid)
        if task not in t2tid:
            t2tid[task] = len(t2tid)
            task_y.append(o2oid[truth])
        else:
            task_y[t2tid[task]] = o2oid[truth]
    data.task_y = torch.tensor(task_y)

    data.num_worker = len(w2wid)
    data.num_task = len(t2tid)
    data.num_option = len(o2oid)

    data.setup()
    return data


def get_dataset_list(cfg):
    data_dir = Path(cfg.path.data)
    dataset_list = [data_path.name for data_path in data_dir.iterdir() if data_path.is_dir()]
    return dataset_list


def run(cfg, selected_dataset=None, **kwargs):
    data_dir = Path(cfg.path.data)
    data_dict = dict()
    for data_path in data_dir.iterdir():
        if selected_dataset is not None and data_path.name != selected_dataset:
            continue
        data_dict[data_path.name] = load_single_data(cfg, data_path).to(cfg.device)
    return data_dict
