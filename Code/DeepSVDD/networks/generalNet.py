import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from DeepSVDD.base.base_net import BaseNet


class GeneralNet(BaseNet):

    def __init__(self):
        super().__init__()
        input_size = 23634
        self.rep_dim = 128
        self.en_coder = nn.Sequential(OrderedDict([
                        # encoder
                        ('en_1', nn.Linear(input_size, int(input_size/2))),
                        ('relu_1', nn.ReLU()),
                        ('en_2', nn.Linear(int(input_size/2), int(input_size/4))),
                        ('relu_2', nn.ReLU()),
                        ('en_3', nn.Linear(int(input_size/4), int(input_size/8))),
                        ('relu_3', nn.ReLU()),
                        ('en_4', nn.Linear(int(input_size/8), self.rep_dim)),
                        ('relu_4', nn.ReLU()),
        ]))

    def forward(self, input):
        latent_emb = self.en_coder(input.view(128, -1))
        return latent_emb


class GeneralNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()
        input_size = 23634
        self.rep_dim = 128
        self.en_coder = nn.Sequential(OrderedDict([
            # encoder
            ('en_1', nn.Linear(input_size, int(input_size / 2))),
            ('relu_1', nn.ReLU()),
            ('en_2', nn.Linear(int(input_size / 2), int(input_size / 4))),
            ('relu_2', nn.ReLU()),
            ('en_3', nn.Linear(int(input_size / 4), int(input_size / 8))),
            ('relu_3', nn.ReLU()),
            ('en_4', nn.Linear(int(input_size / 8), self.rep_dim)),
            ('relu_4', nn.ReLU()),
        ]))
        self.de_coder = nn.Sequential(OrderedDict([
            # decoder
            ('de_1', nn.Linear(self.rep_dim, int(input_size / 8))),
            ('relu_5', nn.ReLU()),
            ('de_2', nn.Linear(int(input_size / 8), int(input_size / 4))),
            ('relu_6', nn.ReLU()),
            ('de_3', nn.Linear(int(input_size / 4), int(input_size / 2))),
            ('relu_7', nn.ReLU()),
            ('de_4', nn.Linear(int(input_size / 2), input_size)),
            ('relu_8', nn.ReLU()),
        ]))


    def forward(self, input):
        # print(input.shape)
        # print(input.view(-1, 606*39).shape)
        latent_emb = self.en_coder(input.view(-1, 606*39))
        out = self.de_coder(latent_emb)

        return out.view(-1, 606, 39)

