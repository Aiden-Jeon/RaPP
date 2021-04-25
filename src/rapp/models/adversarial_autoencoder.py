from typing import Tuple

import torch
import torch.nn as nn

from .autoencoder import AutoEncoder
from ..layer import FullyConnectedLayer
from ..utils import get_hidden_sizes


class AdversarialAutoEncoder(AutoEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 3,
        d_layers: int = 3,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
        )
        decoder_hidden_sizes = get_hidden_sizes(hidden_size, 1, d_layers)
        disc_layers = []
        for i, o in zip(decoder_hidden_sizes[:-2], decoder_hidden_sizes[1:-1]):
            disc_layers += [FullyConnectedLayer(i, o, "leakyrelu")]
        disc_layers += [
            FullyConnectedLayer(
                decoder_hidden_sizes[-2], decoder_hidden_sizes[-1], act="sigmoid"
            )
        ]
        self.discriminator = nn.Sequential(*disc_layers)
        self.bce_loss = nn.BCELoss()
        self.automatic_optimization = False

    def get_recon_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_recon = self.forward(x)
        recon_loss = self.loss_fn(x_recon, x)
        return recon_loss

    def get_D_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_fake = self.encoder(x)
        z_true = torch.randn(z_fake.size()).to(self.device)

        z_true_pred = self.discriminator(z_true)
        z_fake_pred = self.discriminator(z_fake)

        target_ones = torch.ones(x.size(0), 1).to(self.device)
        target_zeros = torch.zeros(x.size(0), 1).to(self.device)

        true_loss = self.bce_loss(z_true_pred, target_ones)
        fake_loss = self.bce_loss(z_fake_pred, target_zeros)

        D_loss = true_loss + fake_loss
        return D_loss

    def get_G_loss_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        target_ones = torch.ones(batch_size, 1).to(self.device)
        z_fake = self.encoder(x)
        z_fake_pred = self.discriminator(z_fake)
        G_loss = self.bce_loss(z_fake_pred, target_ones)
        return G_loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        encoder_opt, decoder_opt, disc_opt = self.optimizers()
        x, y = batch
        x = x.view(x.size(0), -1)
        #
        # update encoder, decoder
        #
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        recon_loss = self.get_recon_loss(x)
        recon_loss.backward()
        decoder_opt.step()
        encoder_opt.step()
        #
        # update discriminator
        #
        disc_opt.zero_grad()
        D_loss = self.get_D_loss(x, y)
        D_loss.backward()
        disc_opt.step()
        #
        # update generator
        #
        encoder_opt.zero_grad()
        G_loss = self.get_G_loss_value(x, y)
        G_loss.backward()
        encoder_opt.step()

        loss = recon_loss + D_loss + G_loss

        self.log_dict({"recon_loss": recon_loss, "D_loss": D_loss, "G_loss": G_loss})
        return loss

    def configure_optimizers(self):
        encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        return encoder_opt, decoder_opt, disc_opt
