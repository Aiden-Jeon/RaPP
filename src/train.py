import argparse
import pytorch_lightning as pl

from rapp.data import MNISTDataModule
from rapp.models import AutoEncoder, RaPP


def main(
    model: str,
    dataset: str,
    target_label: int,
    data_dir: str,
    hidden_size: int,
    n_layers: int,
    max_epochs: int,
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
    else:
        raise ValueError(f"Not valid model name {model}")

    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(auto_encoder, data_module)
    rapp = RaPP(auto_encoder)
    rapp.fit(data_module.train_dataloader())
    result = rapp.test(data_module.test_dataloader())
    auto_encoder.log_dict(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ae")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="/")
    parser.add_argument("--hidden_size", type=int, default=20)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=200)
    args = parser.parse_args()

    main(
        model=args.model,
        dataset=args.dataset,
        target_label=args.target_label,
        data_dir=args.data_dir,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        max_epochs=args.max_epochs,
    )
