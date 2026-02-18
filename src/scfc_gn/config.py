# src/das_classification/config.py 

from __future__ import annotations 

from dataclasses import dataclass 
from pathlib import Path
from typing import Any, Dict
from scfc_gn.ucla.models import FilePattern

import yaml 


@dataclass(frozen=True)
class SplitCfg:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


@dataclass(frozen=True)
class DatasetCfg:
    root: str
    pattern: FilePattern
    split: SplitCfg = SplitCfg()



@dataclass(frozen=True)
class TrainCfg:
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 20




@dataclass(frozen=True)
class RunCfg:
    base_dir: str = "runs"
    name: str = "baseline"
    splits_dir: str = "data/interim/splits"
    device: str = ""  # auto if empty
    seed: int = 42
    deterministic: bool = True


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetCfg
    train: TrainCfg
    run: RunCfg


def _get(d: Dict[str, Any], key: str, default: Any = None):
    return d[key] if key in d else default


def load_config(path: str) -> AppConfig:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ds_raw = raw.get("dataset", {})
    tr_raw = raw.get("training", raw.get("train", {}))  # allow both keys
    run_raw = raw.get("run", {})

    ds_split_raw = ds_raw.get("split", {}) or {}
    split = SplitCfg(
        train=float(_get(ds_split_raw, "train", 0.8)),
        val=float(_get(ds_split_raw, "val", 0.1)),
        test=float(_get(ds_split_raw, "test", 0.1)),
        )

    fp_raw = ds_raw.get("file_pattern", {})
    file_pattern = FilePattern(
        dti_files=str(_get(fp_raw, "", "data/*_DTI_connectivity_matrix_file.txt")),
        fc_files=str(_get(fp_raw, "", "data/*_rsfMRI_connectivity_matrix_file.txt")),
        xyz_files=str(_get(fp_raw, "", "data/*_DTI_region_xyz_centers_file.txt")),
        region_files=str(_get(fp_raw, "", "data/*_DTI_region_names_full_file.txt")),
    )
    
    dataset = DatasetCfg(
        root=_get(ds_raw, "root", "data/raw/DAS-dataset/data"),
        pattern= file_pattern,
        split =split,
    )


    train = TrainCfg(
        batch_size=int(_get(tr_raw, "batch_size", 8)),
        num_workers=int(_get(tr_raw, "num_workers", 4)),
        epochs=int(_get(tr_raw, "epochs", 20)),
        lr=float(_get(tr_raw, "lr", 1e-3)),
        weight_decay=float(_get(tr_raw, "weight_decay", 1e-4)),
        grad_clip=float(_get(tr_raw, "grad_clip", 1.0)),
        log_interval=int(_get(tr_raw, "log_interval", 20)),
    )

    run = RunCfg(
        base_dir=_get(run_raw, "base_dir", "runs"),
        name=_get(run_raw, "name", "baseline"),
        splits_dir=_get(run_raw, "splits_dir", "data/interim/splits"),
        device=_get(run_raw, "device", ""),
        seed=int(_get(run_raw, "seed", 42)),
        deterministic=bool(_get(run_raw, "deterministic", True)),
    )


    cfg = AppConfig(dataset=dataset, train=train, run=run)

    return cfg