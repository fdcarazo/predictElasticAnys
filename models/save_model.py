#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: NN (FFNN) used to train the model in UM's or JeanZay's Cluster-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## start date (Fr): Sun Mar  3 22:48:52 CET 2024-.
## last modify (Fr): -.
##
## ====================================================================== INI79

## ====================================================================== INI79
## 1- include packages, modules, variables, etc.-.
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pickle

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
        ## print(layers_data[0]); input(99)
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
        self.learning_rate=learning_rate
        self.optimizer=eval(optimizer)(params= self.parameters(), lr= learning_rate)
        self.criterion=nn.MSELoss()

    def forward(self, input_data):
        for layer in self.layers:
            input_data=layer(input_data)
        return input_data

    def save_params_model(self,dir_log,bst,bsv,epochs,ds_file):
        ''' method to save the DL model's main params '''
        ## print a loogBookFile-.
        log_file=dir_sve+'/'+ 'log.p'; file_log=open(log_file, 'wb')
        ## print(log_file, type(log_file), sep='\n')
        ## '/gpfswork/rech/irr/uhe87yl/carazof/scratch/fernando/resDL_2_SS/log.p'

        ## dictionary2print-.
        log_dict={'learning_rate': self.learning_rate,
                  'batch_size_train': bst,
                  'batch_size_val': bsv,
                  'num_epochs': epochs,
                  'layers_list': layers_data,
                  'optimizer': self.optimizer,
                  'loss': self.criterion,
                  'dataset_file': ds_file,
                  'folder_out': dir_results,
                  }
        pickle.dump(log_dict, file_log)
        file_log.close()

# - =======================================================================END79

if __name__=='__main__':
    pass
