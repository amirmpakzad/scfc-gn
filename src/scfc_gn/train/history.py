from __future__ import annotations
from dataclasses import dataclass
import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainHistory:
    epochs: list[int]
    train_losses: list[float]
    val_losses: list[float]


class JsonlHistoryWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_dataclass(self, step_info: Dict[str, Any],
                      obj: Any, prefix: str) -> None:
        if is_dataclass(obj):
            d = asdict(obj)
        elif isinstance(obj, dict):
            d = obj
        else:
            raise TypeError('object must be a dict or dataclass')

        rec = dict(step_info)
        for k, v in d.items():
            rec[f"{prefix}_{k}"] = v
        self.log(rec)


def get_history(path: str):
    epochs = []
    train_losses = []
    val_losses = []

    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record["type"] == "epoch":
                epochs.append(record["epoch"])
                train_losses.append(record["train_loss"])
                val_losses.append(record["val_loss"])

    return TrainHistory(epochs, train_losses, val_losses)