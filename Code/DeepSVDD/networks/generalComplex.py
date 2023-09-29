import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from collections import OrderedDict

import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd


from DeepSVDD.base.base_net import BaseNet
from DeepSVDD.networks.util.torchTIMNET import TIMNET

device = torch.device('cuda:0')


class GeneralComplex(BaseNet):
    def __init__(self, input_shape,
                 filter_size=39, kernel_size=2, stack_size=1, dilation_size=10, dropout=0.1):
        super(GeneralComplex, self).__init__()
        self.data_shape = input_shape
        print("TIMNET MODEL SHAPE-DeepSVDD:", input_shape)

        self.rep_dim = dilation_size * input_shape[2]

        self.tim = TIMNET(
            input_shape=[self.data_shape[2], self.data_shape[1]],
            nb_filters=filter_size,
            kernel_size=kernel_size,
            nb_stacks=stack_size,
            dilations=dilation_size,
            dropout_rate=dropout,
            activation='relu',
            return_sequences=True,
            name='TIMNET')

    # return number of features * dilation_size
    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        multi_decisions = self.tim(x)

        return torch.flatten(multi_decisions, start_dim=1)


class Conv_Decoder(BaseNet):

    def __init__(self, num_of_feature, in_size, out_size, kernel_size, dropout):
        super(Conv_Decoder, self).__init__()

        self.in_size = in_size
        self.conv_1 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv1d(in_channels=in_size, out_channels=100, kernel_size=kernel_size)),
            ('bn_1', nn.BatchNorm1d(num_features=100, affine=True)),
            ('relu_1', nn.ReLU()),
            ('drop_1', nn.Dropout1d(p=dropout)),
        ]))

        self.conv_2 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv1d(in_channels=100, out_channels=200, kernel_size=kernel_size)),
            ('bn_1', nn.BatchNorm1d(num_features=200, affine=True)),
            ('relu_1', nn.ReLU()),
            ('drop_1', nn.Dropout1d(p=dropout)),
        ]))

        self.conv_3 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv1d(in_channels=200, out_channels=400, kernel_size=kernel_size)),
            ('bn_1', nn.BatchNorm1d(num_features=400, affine=True)),
            ('relu_1', nn.ReLU()),
            ('drop_1', nn.Dropout1d(p=dropout)),
        ]))

        self.conv_4 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv1d(in_channels=400, out_channels=out_size, kernel_size=kernel_size)),
            ('bn_1', nn.BatchNorm1d(num_features=out_size, affine=True)),
            ('relu_1', nn.ReLU()),
            ('drop_1', nn.Dropout1d(p=dropout)),
        ]))

    # input: batch size * encoder output
    # return: batch size * frames number * num of features
    def forward(self, input):
        input = torch.split(input, split_size_or_sections=self.in_size, dim=1)
        input = torch.stack(input, dim=1)
        input = torch.permute(input, (0, 2, 1))
        r = self.conv_1(input)
        r = self.conv_2(r)
        r = self.conv_3(r)
        r = self.conv_4(r)

        return r


class LinearExpansion(nn.Module):

    def __init__(self, num_of_feature, in_size, out_size, dropout):
        super(LinearExpansion, self).__init__()

        self.linears = nn.ModuleList()

        for i in range(num_of_feature):
            linear = nn.Sequential(OrderedDict([
                ('linear_1', nn.Linear(in_size, out_size)),
                ('bn_1', nn.BatchNorm1d(num_features=out_size, affine=True)),
                ('relu_1', nn.ReLU()),
                ('drop_1', nn.Dropout1d(p=dropout)),
            ]))

            self.linears.append(linear)

    # return: batch size *  num of features * in_size
    def forward(self, input):

        result = []

        for i in range(input.shape[1]):
            input_i = input[:, i, :]
            result.append(self.linears[i].forward(input_i))

        result = torch.stack(result)

        result = torch.permute(result, (1, 0, 2))
        return result


class Linear_Decoder(BaseNet):
    def __init__(self, num_of_feature, in_size, out_size, dropout):
        super(Linear_Decoder, self).__init__()

        self.in_size = in_size
        self.l1 = LinearExpansion(num_of_feature, in_size, 100, dropout)
        self.l2 = LinearExpansion(num_of_feature, 100, 200, dropout)
        self.l3 = LinearExpansion(num_of_feature, 200, 400, dropout)
        self.l4 = LinearExpansion(num_of_feature, 400, out_size, dropout)

        # input: batch size * encoder output

    # return: batch size * frames number * num of features
    def forward(self, input):
        input = torch.split(input, split_size_or_sections=self.in_size, dim=1)
        input = torch.stack(input, dim=1)

        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)

        out = torch.permute(out, (0, 2, 1))
        return out


class Linear_Autoencoder(BaseNet):
    def __init__(self, input_shape,
                 filter_size=39, kernel_size=2, stack_size=1, dilation_size=10, dropout=0.1
                 ):
        super(Linear_Autoencoder, self).__init__()
        self.rep_dim = dilation_size * input_shape[2]

        self.encoder = GeneralComplex(input_shape, filter_size, kernel_size, stack_size, dilation_size, dropout)

        self.decoder = Linear_Decoder(input_shape[2], dilation_size, input_shape[1], dropout)

    def forward(self, x):
        latent_space = self.encoder(x)
        return self.decoder(latent_space)


class Conv_Autoencoder(BaseNet):
    def __init__(self, input_shape,
                 filter_size=39, kernel_size=2, stack_size=1, dilation_size=10, dropout=0.1
                 ):
        super(Conv_Autoencoder, self).__init__()
        self.rep_dim = dilation_size * input_shape[2]

        self.encoder = GeneralComplex(input_shape, filter_size, kernel_size, stack_size, dilation_size, dropout)

        self.decoder = Conv_Decoder(input_shape[2], dilation_size, input_shape[1], 1, dropout)

    def forward(self, x):
        latent_space = self.encoder(x)
        return self.decoder(latent_space)


if __name__ == "__main__":
    x = torch.rand(5, 188, 39)

    l1 = Linear_Autoencoder(x.shape, 39, 2, 1, 10, 0.1)
    c1 = Conv_Autoencoder(x.shape, 39, 2, 1, 10, 0.1)
    y1 = l1.forward(x)
    y2 = c1.forward(x)