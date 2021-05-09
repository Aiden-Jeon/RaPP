from logging import log
from typing import Tuple

import torch
import torch.nn as nn

from .autoencoder import AutoEncoder
from ..layer import FullyConnectedLayer
from ..utils import get_hidden_sizes


class VariationalAutoEncoder(AutoEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        k: int = 10,
        loss_reduction: str = "sum",
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            loss_reduction=loss_reduction,
        )
        encoder_hidden_sizes = get_hidden_sizes(input_size, hidden_size * 2, n_layers)
        encoder_layers = []
        for i, o in zip(encoder_hidden_sizes[:-2], encoder_hidden_sizes[1:-1]):
            encoder_layers += [FullyConnectedLayer(i, o, "leakyrelu")]
        encoder_layers += [
            FullyConnectedLayer(
                encoder_hidden_sizes[-2], encoder_hidden_sizes[-1], act=None
            )
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        self.k = k
        self.kld_loss_fn = lambda mu, logvar: 0.5 * (
            mu ** 2 + logvar.exp() - logvar - 1
        )
        self.loss_reduction = loss_reduction

    def reparameterize_normal(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        expanded_sigma = sigma.unsqueeze(0).expand(k, *sigma.size())
        z = torch.randn_like(expanded_sigma).mul(expanded_sigma) + mu
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        mu, logvar = z.split(z.size(-1) // 2, dim=-1)
        z = self.reparameterize_normal(mu, logvar, self.k)
        return {"z": z, "mu": mu, "logvar": logvar}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        k, batch_size, hidden_size = z.size()
        x_hat = self.decoder(z.view(-1, hidden_size))
        return x_hat.view(k, batch_size, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encode_dict = self.encode(x)
        recon_x = self.decode(encode_dict["z"]).mean(dim=0)
        return recon_x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        encode_dict = self.encode(x)
        recon_x = self.decode(encode_dict["z"])
        _x = x.unsqueeze(0).expand(self.k, *x.size()).contiguous()
        recon_loss = self.loss_fn(recon_x, _x)

        mu = encode_dict["mu"]
        logvar = encode_dict["logvar"]
        kld_loss = self.kld_loss_fn(mu, logvar)

        if self.loss_reduction == "mean":
            kld_loss = kld_loss.mean()
            recon_loss /= self.k
        else:
            kld_loss = kld_loss.sum()
            recon_loss *= x.size(1)
            kld_loss *= encode_dict["z"].size(1)

        loss = recon_loss + kld_loss
        self.log_dict({"recon_loss": recon_loss, "kld_loss": kld_loss})
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        recon_x = self(x)
        loss = self.loss_fn(x, recon_x)
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
