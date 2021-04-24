from typing import Tuple, Dict
import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..layer import FullyConnectedLayer
from ..utils import get_hidden_sizes
from ..metrics import get_auroc, get_aupr


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        encoder_hidden_sizes = get_hidden_sizes(input_size, hidden_size, n_layers)
        encoder_layers = []
        for i, o in zip(encoder_hidden_sizes[:-2], encoder_hidden_sizes[1:-1]):
            encoder_layers += [FullyConnectedLayer(i, o, "leakyrelu")]
        encoder_layers += [
            FullyConnectedLayer(
                encoder_hidden_sizes[-2], encoder_hidden_sizes[-1], act=None,
            )
        ]
        decoder_hidden_sizes = get_hidden_sizes(hidden_size, input_size, n_layers)
        decoder_layers = []
        for i, o in zip(decoder_hidden_sizes[:-2], decoder_hidden_sizes[1:-1]):
            decoder_layers += [FullyConnectedLayer(i, o, "leakyrelu")]
        decoder_layers += [
            FullyConnectedLayer(
                decoder_hidden_sizes[-2], decoder_hidden_sizes[-1], act=None,
            )
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(X))

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        recon_x = self(x)
        loss = self.loss_fn(x, recon_x)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        recon_x = self(x)
        loss = self.loss_fn(x, recon_x)
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        recon_x = self(x)
        recon_err = ((recon_x - x) ** 2).mean(axis=1)
        return {"score": recon_err, "label": y}

    def test_epoch_end(self, outputs: Dict[torch.Tensor, torch.Tensor]):
        score = []
        label = []
        for output in outputs:
            score += [output["score"]]
            label += [output["label"]]
        score = torch.cat(score).numpy()
        label = torch.cat(label).numpy()
        auroc = get_auroc(label, score)
        aupr = get_aupr(label, score)
        self.log("auroc", auroc)
        self.log("aupr", aupr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
