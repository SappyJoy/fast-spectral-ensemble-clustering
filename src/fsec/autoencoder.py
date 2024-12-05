import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, encoding_dim=64, hidden_dims=[128, 64]):
        super(Autoencoder, self).__init__()
        # Encoder
        encoder_layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            last_dim = h_dim
        encoder_layers.append(nn.Linear(last_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        last_dim = encoding_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            last_dim = h_dim
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

