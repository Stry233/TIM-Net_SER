import torch
import torch.nn as nn 
import torch.nn.functional as F
from TIMNET import TIMNET
import argparse
from collections import OrderedDict

import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import pandas as pd

from TIMNET import TIMNET
import argparse

import xlwt
from xlwt import Workbook
import torch.optim as optim
import os
import numpy as np


from DeepSVDD.base.base_net import BaseNet


device = torch.device('cuda:0')

 

class WeightLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(WeightLayer, self).__init__()
        self.kernel = nn.Parameter(torch.rand(input_size, output_size))
        torch.nn.init.normal_(self.kernel,mean=0, std=1.0)

    def forward(self, x):
        #tempx = torch.permute(x, (0,2,1))
        x = torch.matmul(x,self.kernel)
        x = torch.squeeze(x,-1)
        return  x

class TIMNET_Encoder(BaseNet):
    def __init__(self,input_shape,
                 filter_size,kernel_size,stack_size,dilation_size,dropout):
        super(TIMNET_Encoder, self).__init__()
        self.data_shape = input_shape
        print("TIMNET MODEL SHAPE:",input_shape)


        self.tim = TIMNET(
                        input_shape = [self.data_shape[2], self.data_shape[1]],
                        nb_filters=filter_size,
                        kernel_size=kernel_size, 
                        nb_stacks=stack_size,
                        dilations=dilation_size,
                        dropout_rate=dropout,
                        activation = 'relu',
                        return_sequences=True, 
                        name='TIMNET')
      

    
    # return number of features * dilation_size
    def forward(self, x):
        x = torch.permute(x, (0,2,1))
        multi_decisions = self.tim(x)
        
        return torch.flatten(multi_decisions, start_dim=1)



class Conv_Decoder(BaseNet):
    
    def __init__(self, num_of_feature, in_size, out_size,kernel_size, dropout):
        super(Conv_Decoder, self).__init__()
        
        self.in_size = in_size
        self.conv_1 = nn.Sequential(OrderedDict([
                    ('conv_1', nn.Conv1d(in_channels=in_size, out_channels=100, kernel_size =kernel_size)),
                    ('bn_1', nn.BatchNorm1d(num_features=100,affine=True)),
                    ('relu_1', nn.ReLU()),
                    ('drop_1',nn.Dropout1d(p=dropout)),
            ]))

        self.conv_2 = nn.Sequential(OrderedDict([
                    ('conv_1', nn.Conv1d(in_channels=100, out_channels=200, kernel_size =kernel_size)),
                    ('bn_1', nn.BatchNorm1d(num_features=200,affine=True)),
                    ('relu_1', nn.ReLU()),
                    ('drop_1',nn.Dropout1d(p=dropout)),
            ]))
        
        self.conv_3 = nn.Sequential(OrderedDict([
                    ('conv_1', nn.Conv1d(in_channels=200, out_channels=400, kernel_size =kernel_size)),
                    ('bn_1', nn.BatchNorm1d(num_features=400,affine=True)),
                    ('relu_1', nn.ReLU()),
                    ('drop_1',nn.Dropout1d(p=dropout)),
            ]))
        
        self.conv_4 = nn.Sequential(OrderedDict([
                    ('conv_1', nn.Conv1d(in_channels=400, out_channels=out_size, kernel_size =kernel_size)),
                    ('bn_1', nn.BatchNorm1d(num_features=out_size,affine=True)),
                    ('relu_1', nn.ReLU()),
                    ('drop_1',nn.Dropout1d(p=dropout)),
            ]))
        
    # input: batch size * encoder output
    # return: batch size * frames number * num of features
    def forward(self, input):
        input = torch.split(input, split_size_or_sections= self.in_size, dim=1)
        input = torch.stack(input, dim=1)
        input= torch.permute(input, (0,2,1))
        r = self.conv_1(input)
        r=  self.conv_2(r)
        r = self.conv_3(r)
        r= self.conv_4(r)

        return r


class LinearExpansion(nn.Module):
    
    def __init__(self, num_of_feature, in_size, out_size, dropout):
        super(LinearExpansion, self).__init__()
        
        self.linears = nn.ModuleList()
        
        for i in range(num_of_feature):
            linear = nn.Sequential(OrderedDict([
                    ('linear_1', nn.Linear(in_size, out_size)),
                    ('bn_1', nn.BatchNorm1d(num_features=out_size,affine=True)),
                    ('relu_1', nn.ReLU()),
                    ('drop_1',nn.Dropout1d(p=dropout)),
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
        self.l1 = LinearExpansion(num_of_feature,in_size, 100, dropout) 
        self.l2 = LinearExpansion(num_of_feature,100, 200, dropout) 
        self.l3 = LinearExpansion(num_of_feature,200, 400, dropout) 
        self.l4 = LinearExpansion(num_of_feature,400, out_size, dropout) 
    
    # input: batch size * encoder output
    # return: batch size * frames number * num of features
    def forward(self, input):
        input = torch.split(input, split_size_or_sections= self.in_size, dim=1)
        input = torch.stack(input, dim=1)

        out = self.l1(input)
        out = self.l2 (out)
        out = self.l3(out)
        out = self.l4(out)

        out = torch.permute(out, (0, 2, 1))
        return out
    
if __name__ == "__main__":
    x = torch.rand(5, 606, 39)

    tim = TIMNET_Encoder(x.shape, 39, 2, 1, 10, 0.1)
    de1 = Linear_Decoder(39, 10, 606, 0.1)
    de2 = Conv_Decoder(39, 10, 606,1, 0.1)
    y = tim.forward(x)

    # input = torch.split(y, split_size_or_sections= 10, dim=1)
    # input = torch.stack(input, dim=1)

    x1 = de1.forward(y)
    x2 = de2.forward(y)
    print(y)
