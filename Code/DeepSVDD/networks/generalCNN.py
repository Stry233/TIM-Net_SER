import torch.nn as nn
from collections import OrderedDict
from DeepSVDD.base.base_net import BaseNet


class GeneralCNN(BaseNet):

    def __init__(self):
        super().__init__()

        # Assuming the data is of shape [batch_size, 606, 39]
        # We can treat it as [batch_size, 1, 606, 39] for convolutional layers
        self.rep_dim = 288

        self.encoder = nn.Sequential(
            nn.Conv1d(606, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, input):
        # Reshape the input to [batch_size, 1, 606, 39] for convolutional layers
        latent_emb = self.encoder(input)
        return latent_emb.view(latent_emb.size(0), -1)


class GeneralCNN_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        # Assuming the data is of shape [batch_size, 606, 39]
        # We can treat it as [batch_size, 1, 606, 39] for convolutional layers
        self.rep_dim = 128

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(606, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=2, stride=2, padding=0, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 606, kernel_size=2, stride=2, padding=0, output_padding=1),
            nn.ReLU()
        )

    def forward(self, input):
        # Reshape the input to [batch_size, 1, 606, 39] for convolutional layers
        latent_emb = self.encoder(input)
        # print(latent_emb.shape)
        out = self.decoder(latent_emb)
        return out
