import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import typer
from .config import load_config
from scfc_gn.utils.loger import setup_logger, make_run_dir
from scfc_gn.networks.gcn_model import GCNModel, ModelConfig
from scfc_gn.train.loop import train_loop, TrainConfig
from scfc_gn.ucla.load_data import get_all_subjects
from scfc_gn.ucla.models import Subject
from scfc_gn.utils.graph import edge_index_from_connectivity, ConversionConfig, EdgeIndexResult

app = typer.Typer(no_args_is_help=True)


@app.command()
def train(
    config: str = typer.Option(
        "configs/app.yaml",
        help="Path to an app config YAML",
    ),
):
    cfg = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        edge_cfg = ConversionConfig(
            directed=False, 
            include_self_loops=False, 
            threshold=1e-6)
        edges = edge_index_from_connectivity(sc, config=edge_cfg)
        data = Data(
            x=I.float(),
            edge_index=edges.edge_index.long(),
            edge_attr=edges.edge_attr.float(),
            y=fc.float())

        data_list.append(data)



    #get loader 
    dl = DataLoader(data_list, batch_size=cfg.train.batch_size, shuffle=True)
    model = GCNModel(ModelConfig(in_channels=n))
    #train loop
    train_cfg = TrainConfig(
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        grad_clip=cfg.train.grad_clip,
        log_interval=cfg.train.log_interval,
        save_dir=str(run_dir)
    )
    train_loop(model, dl, device, train_cfg)
    pass 


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