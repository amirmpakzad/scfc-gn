from datetime import datetime
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import typer
from .config import load_config
from scfc_gn.utils.loger import setup_logger, make_run_dir
from scfc_gn.networks.gcn_model import GCNModel, ModelConfig
from scfc_gn.train.loop import train_loop, TrainConfig, TrainResult
from scfc_gn.train.history import get_history
from scfc_gn.ucla.load_data import get_all_subjects
from scfc_gn.ucla.models import Subject
from scfc_gn.data.dataset import SCFCDataset
from scfc_gn.utils.graph import edge_index_from_connectivity, ConversionConfig, to_degree_matrix
from scfc_gn.viz.plot_history import plot_history
from scfc_gn.viz.plot_matrices import show_as_image
from scfc_gn.train.test import load_checkpoint, test_loop

app = typer.Typer(no_args_is_help=True)


@app.command()
def train(
    config: str = typer.Option(
        "configs/app.yaml",
        help="Path to an app config YAML",
    ),
):
    cfg = load_config(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir(cfg.run.base_dir, name=cfg.run.name)
    logger = setup_logger(run_dir)
    #load data
    subs, _ = get_all_subjects(cfg.dataset.pattern)
    if not subs:
        typer.echo("No subjects found")
        raise typer.Exit(code=1)

    data_list = []
    n = len(subs[0].whole_data.matrices.sc_matrix)
    for item in subs:
        sc = torch.from_numpy(item.whole_data.matrices.sc_matrix)
        fc = torch.from_numpy(item.whole_data.matrices.fc_matrix)

        I = torch.eye(n)
        x = to_degree_matrix(sc, threshold=0.01)
        edge_cfg = ConversionConfig(
            directed=False, 
            include_self_loops=False, 
            threshold=1e-6)
        edges = edge_index_from_connectivity(sc, config=edge_cfg)
        data = Data(
            x=sc.float(),
            edge_index=edges.edge_index.long(),
            edge_attr=edges.edge_attr.float(),
            y=fc.float())

        data_list.append(data)

    ds = SCFCDataset(data_list)

    #split
    n_ds = len(ds)
    n_train = int(0.8 * n_ds)
    n_val = int(0.1 * n_ds)
    n_test = n_ds - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        ds,
        [n_train, n_val, n_test],
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=True)

    #get loader 
    dl = DataLoader(data_list, batch_size=cfg.train.batch_size, shuffle=True)
    model = GCNModel(ModelConfig(in_channels=n, dropout=0.1))
    #train loop
    train_cfg = TrainConfig(
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        grad_clip=cfg.train.grad_clip,
        log_interval=cfg.train.log_interval,
        save_dir=str(run_dir)
    )
    result = train_loop(
        model=model,
        cfg=train_cfg,
        train_loader=train_loader,
        device=device,
        val_loader=val_loader
    )

    history = get_history(result.dir)

    plot_history(
        history=history,
        pred_std=result.last_pred_std,
        pred_mean=result.last_pred_mean,
        y_mean=result.y_mean,
        y_std=result.y_std,
        corr=result.pearson_corr,
        baseline=0.13
    )


    # --- load checkpoint ---
    ckpt_path = result.ckpt_path
    epoch = load_checkpoint(model, str(ckpt_path), device=device)

    with torch.inference_mode():
        test_result = test_loop(model, test_loader, device=device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    show_as_image(test_result.last_y, title="FC", timestamp = timestamp)
    show_as_image(test_result.last_pred, title="Predicted FC", timestamp = timestamp)



@app.command()
def test():
    pass


@app.command()
def get_subjects(
    config: str = typer.Option(
        "configs/app.yaml",
        help="Path to an app config YAML",
    ),
):
    cfg = load_config(config)
    subs, _ = get_all_subjects(cfg.dataset.pattern)
    if not subs:
        print("No subject found")
        return 
    print(f"Total Subjects: {len(subs)}")


if __name__ == "__main__":
    app()