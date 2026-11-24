from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .data.classification_io import load_dataset as load_classification
from .data.simulation import ClassificationSimulator
from .model.classification import ClassificationModel

ConfigValidator = Callable[[dict], None]


def _load_classification(path, config):
    return load_classification(path, dim=config["model"]["dim"])


def _validate_classification(config):
    if "num_option_range" not in config["simulator"]:
        raise ValueError("classification configuration requires simulator.num_option_range")
    options = config["simulator"]["num_option_range"]
    if not isinstance(options, (list, tuple)) or len(options) != 2:
        raise ValueError("simulator.num_option_range must contain two integers")
    lower, upper = options
    if type(lower) is not int or type(upper) is not int or lower < 2 or lower > upper:
        raise ValueError("simulator.num_option_range must be increasing integers >= 2")


@dataclass(frozen=True)
class CrowdTask:
    """A pluggable crowd-aggregation task.

    Add a new task by implementing a model, a simulator, and a data loader,
    then registering a ``CrowdTask`` at the bottom of this module. The training
    and the entry points depend only on this interface.
    """

    name: str
    model_class: type
    simulator_class: type
    loader: Callable
    validate: ConfigValidator | None = None
    needs_model_dim: bool = False

    def build_model(self, config, device):
        return self.model_class(**config["model"]).to(device)

    def build_simulator(self, config):
        options = dict(config["simulator"])
        if self.needs_model_dim:
            options["dim"] = config["model"]["dim"]
        return self.simulator_class(**options)

    def load(self, config, path):
        return self.loader(Path(path), config)


_TASKS: dict[str, CrowdTask] = {}


def register_task(task: CrowdTask) -> CrowdTask:
    if task.name in _TASKS:
        raise ValueError(f"task {task.name!r} is already registered")
    _TASKS[task.name] = task
    return task


def available_tasks():
    return tuple(_TASKS)


def get_task(name):
    try:
        return _TASKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown task {name!r}; choose from {available_tasks()}") from exc


def dataset_paths(root):
    if root is None:
        return {}
    root = Path(root)
    if (root / "label.csv").is_file() and (root / "truth.csv").is_file():
        return {root.name: root}
    paths = {
        path.name: path for path in sorted(root.iterdir())
        if path.is_dir() and (path / "label.csv").is_file() and (path / "truth.csv").is_file()
    }
    if not paths:
        raise ValueError(f"no label.csv / truth.csv dataset pairs in {root}")
    return paths


register_task(CrowdTask(
    name="classification",
    model_class=ClassificationModel,
    simulator_class=ClassificationSimulator,
    loader=_load_classification,
    validate=_validate_classification,
    needs_model_dim=True,
))
