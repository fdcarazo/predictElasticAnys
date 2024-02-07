#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
##
## SCRIPT: NN (FFNN) used to train the model in UM's or JeanZay's Cluster-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## last_modify: Wed Jan 31 11:42:47 CET 2024-.
##

## ====================================================================== INI79
## 1- include packages, modules, variables, etc.-.
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F
## - ======================================================================END79

## - ======================================================================INI79
## 2- DL class-.
class MLP(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 layers_data: list,
                 device='cpu',
                 learning_rate=0.01,
                 optimizer=optim.Adam
                 ):
        super().__init__()

        self.layers= nn.ModuleList()
        self.input_size= input_size  # can be useful later -.
        self.output_size= output_size  # can be useful later -.

        ## layer_data ==> list of tuples with size of layers and activation-.
        for size, activation, dropout in layers_data: 
            self.layers.append(nn.Linear(input_size, size))
            input_size= size  # for the next layer-.
            ## @ activationLayer-.
            if activation is not None:
                assert isinstance(activation, nn.Module), \
                    'Each tuples should contain a size (int) and a torch.nn.modules.Module.'
                self.layers.append(activation)
            ## @ DropoutLayer-.
            if dropout is not None:
                assert isinstance(dropout, nn.Module), \
                    'Each tuples should contain a size (int) and a torch.nn.modules.Module.'
                self.layers.append(dropout)


        ## self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ## self.to(self.device)
        self.to(device)
        self.learning_rate= learning_rate
        self.optimizer= optimizer(params= self.parameters(), lr= learning_rate)
        self.criterion= nn.MSELoss()

    def forward(self, input_data):
        for layer in self.layers:
            input_data= layer(input_data)
        return input_data
# - =======================================================================END79
