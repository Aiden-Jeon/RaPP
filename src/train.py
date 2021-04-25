import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from rapp.data import MNISTDataModule
from rapp.models import AutoEncoder, RaPP
from rapp.models.adversarial_autoencoder import AdversarialAutoEncoder
from rapp.models.variational_autoencoder import VariationalAutoEncoder


def main(
    model: str,
    dataset: str,
    target_label: int,
    data_dir: str,
    hidden_size: int,
    n_layers: int,
    max_epochs: int,
    experiment_name: str,
    tracking_uri: str,
    n_trial: int,
):
    if dataset == "mnist":
        data_module = MNISTDataModule(
            data_dir=data_dir,
            unseen_label=target_label,
            normalize=True,
        )
        input_size = 28 ** 2
    else:
        raise ValueError(f"Not valid dataset name {dataset}")
    if model == "ae":
        auto_encoder = AutoEncoder(
            input_size=input_size, hidden_size=hidden_size, n_layers=n_layers
        )
    elif model == "vae":
        auto_encoder = VariationalAutoEncoder(
            input_size=input_size, hidden_size=hidden_size, n_layers=n_layers
        )
    elif model == "aae":
        auto_encoder = AdversarialAutoEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            d_layers=n_layers,
        )
    else:
        raise ValueError(f"Not valid model name {model}")
    logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri)
    logger.log_hyperparams(
        {
            "model": model,
            "dataset": dataset,
            "target_label": target_label,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "max_epochs": max_epochs,
            "n_trial": n_trial,
        }
    )
    gpus = -1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(logger=logger, max_epochs=max_epochs, gpus=gpus)
    trainer.fit(auto_encoder, data_module)
    rapp = RaPP(auto_encoder)
    rapp.fit(data_module.train_dataloader())
    result = rapp.test(data_module.test_dataloader())
    logger.log_metrics(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ae")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="/")
    parser.add_argument("--hidden_size", type=int, default=20)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--experiment_name", type=str, default="RaPP")
    parser.add_argument("--tracking_uri", type=str, default="file:./mlruns")
    parser.add_argument("--n_trial", type=int, default=0)
    args = parser.parse_args()

    main(
        model=args.model,
        dataset=args.dataset,
        target_label=args.target_label,
        data_dir=args.data_dir,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        max_epochs=args.max_epochs,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        n_trial=args.n_trial,
    )
