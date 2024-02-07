#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
##
## SCRIPT: NN (Autoencoder) used to train the model in UM's or
##            JeanZay's Cluster-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## last_modify: Wed Jan 31 11:45:18 CET 2024-.
##

## ====================================================================== INI79
## 1- include packages, modules, variables, etc.-.
import torch
import torch.nn as nn
import torch.optim as optim
## - ======================================================================END79

## - ======================================================================INI79
## 2- DL class-.
class MLP_transf(torch.nn.Module):
    def __init__(self, input_size, output_size,
                 device='cpu',
                 learning_rate=0.01,
                 optimizer=optim.Adam):
        super().__init__()
         
        ## Building an linear encoder with Linear
        ## layer followed by Relu activation function
        ## 25 ==> 9 (9 neurons in latent space: from PCA analysis)-.
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )
         
        ## Building an linear decoder with Linear
        ## layer followed by Relu activation function-.
        # 9 ==> 21 (9 neurons in latent space: from PCA analysis)-.
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_size),
        )

        ## self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ## self.to(self.device)
        self.to(device)
        self.learning_rate= learning_rate
        self.optimizer= optimizer(params=self.parameters(), lr=learning_rate)
        self.criterion= nn.MSELoss()

    def forward(self, x):
        encoded= self.encoder(x)
        decoded= self.decoder(encoded)
        return decoded
# - =======================================================================END79
