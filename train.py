import glob
import json
import os
from pathlib import Path
from pprint import pformat, pprint

import dlwheel
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cfm.data import load_data
from cfm.data.crowd_dataset import CrowdDataset
from cfm.model.CFM import CFM


def save_checkpoint(cfg, epoch, model, optimizer, save_path):
    if not cfg.backup:
        return

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    checkpoint_path = save_path / f"{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)

    latest_info = {"latest_checkpoint": str(checkpoint_path)}
    with open(save_path / "latest.json", "w") as f:
        json.dump(latest_info, f)


def load_checkpoint(cfg, model, optimizer, resume_path=None):
    if resume_path is None:
        save_path = Path(cfg.path.log) / cfg.name / "checkpoints"
        latest_file = save_path / "latest.json"

        if latest_file.exists():
            with open(latest_file, "r") as f:
                latest_info = json.load(f)
            resume_path = latest_info["latest_checkpoint"]
        else:
            checkpoint_files = glob.glob(str(save_path / "*.pt"))
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(Path(x).stem))
                resume_path = checkpoint_files[-1]

    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=cfg.device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resumed from epoch {start_epoch}")
        return start_epoch

    return 1


def main():
    cfg = dlwheel.setup()
    pprint(cfg)

    writer = None
    save_path = None

    if cfg.backup:
        writer_path = Path(cfg.path.log) / cfg.name / "run"
        writer_path.mkdir(parents=True, exist_ok=True)

        save_path = Path(cfg.path.log) / cfg.name / "checkpoints"
        save_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(
        CrowdDataset(**cfg.simulator),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: x,
    )
    data_iter = iter(dataloader)

    model = CFM(**cfg.model.to_dict()).to(cfg.device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.resume:
        start_epoch = load_checkpoint(cfg, model, optimizer, cfg.resume_path)
    else:
        start_epoch = 1

    if cfg.backup:
        writer = SummaryWriter(writer_path)

    test_data_dict = load_data.run(cfg)

    with tqdm(range(start_epoch, cfg.epochs), initial=start_epoch, total=cfg.epochs, dynamic_ncols=True) as bar:
        for ep in bar:
            epoch_output = dict()

            # train
            model.train()
            train_data_list = next(data_iter)
            train_data_list = [data.to(cfg.device) for data in train_data_list]
            loss = model.batch_loss(train_data_list) / cfg.gradient_accumulation_steps

            loss.backward()
            if ep % cfg.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_output["train"] = dict()
            epoch_output["train"]["loss"] = loss.item()

            # eval
            model.eval()
            eval_data_list = next(data_iter)
            eval_data_list = [data.to(cfg.device) for data in eval_data_list]
            _, eval_list = model.batch_eval(eval_data_list)
            epoch_output["eval"] = {k: sum(d[k] for d in eval_list) / len(eval_list) for k in eval_list[0]}

            if ep % cfg.test_interval == 0 or ep == cfg.epochs:
                model.eval()
                epoch_output["test"] = dict()
                for dataset_name, data in test_data_dict.items():
                    epoch_output["test"][dataset_name] = model.evaluate(data)

                # output
                tqdm.write(pformat(epoch_output["test"]))
                tqdm.write(pformat(epoch_output["eval"]))
                tqdm.write("*" * 42)

            bar.set_postfix(epoch_output["train"])

            if cfg.backup:
                if ep % cfg.save_interval == 0 or ep == cfg.epochs:
                    save_checkpoint(cfg, ep, model, optimizer, save_path)

                for key, value in epoch_output["train"].items():
                    writer.add_scalar(f"#Train/{key}", value, ep)

                for key, value in epoch_output["eval"].items():
                    writer.add_scalar(f"#Eval/{key}", value, ep)

                if "test" in epoch_output:
                    avg_dict = dict()
                    for dataset_name, data in epoch_output["test"].items():
                        for key, value in data.items():
                            writer.add_scalar(f"{dataset_name}/{key}", value, ep)
                            if key not in avg_dict:
                                avg_dict[key] = []
                            avg_dict[key].append(value)
                    for key, value in avg_dict.items():
                        writer.add_scalar(f"Average/{key}", sum(value) / len(value), ep)


if __name__ == "__main__":
    main()
