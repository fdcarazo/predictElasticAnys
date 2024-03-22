#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: NN (MAML -Meta Agnostic Machine Learning)) used to train the model
##         in UM's or JeanZay's Cluster-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## start date (Fr): Mon Mar  4 22:57:00 CET 2024-.
## last modify (Fr): -.
##
## ====================================================================== INI79

## ====================================================================== INI79
## 1- include packages, modules, variables, etc.-.
import torch
import torch.nn as nn
from torchmeta.modules import MetaModule
## from torchmeta.modules.utils import get_subdict

import torch.optim as optim
import pickle as pickle

## 2don't have problems between float datatype of torch and bnnNN
## if torch.is_tensor(xx) else torc.tensor(xx,dtype=float) (float64) and float
## of NN (float32)-.
torch.set_default_dtype(torch.float64)
## torch.set_default_dtype(torch.float32)
## - ======================================================================END79

## - ======================================================================INI79
## Bayesian NN using torchbnn package-.
class MamlNet(MetaModule):
    def __init__(self, input_size,
                 output_size,
                 layers_data: list,
                 device='cpu',
                 learning_rate=1.0e-2,
                 optimizer=optim.Adam,
                 loss=nn.L1Loss(),
                 weight_decay=1.0e-2,
                 momentum=0.99
                 ):        
        super(MamlNet, self).__init__()
    
        self.layers=nn.ModuleList()
        self.input_size=input_size  # can be useful later -.
        self.output_size=output_size  # can be useful later -.
        self.layers_data=layers_data  # can be useful later -.
        
        ## layer_data ==> list of tuples with size of layers and activation-.
        ## print(layers_data[0]); input(99)
        for size, activation, dropout in layers_data: 
            self.layers.append(nn.Linear(input_size, size))
            input_size=size  # for the next layer-.
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
        self.optimizer=optimizer(params= self.parameters(), lr= learning_rate)
        self.loss=loss # nn.MSELoss()
        self.wd=weight_decay
        self.momentum=momentum
        
    def forward(self,inputs,params=None):
        ## extract parameters-.
        if params is None:
            params=self.parameters()
        else:
            print(parameters); input(11)
            ## params=get_subdict(params,self.named_parameters())
            params=self.get_subdict(params,self.named_parameters())
            print(params); input(11)
        for layer in self.layers: inputs=layer(inputs)
        return inputs

    def save_params_model(self,dir_logs,bst,bsv,epochs,optimizer,loss,ds_file,dir_res):
        ''' method to save the DL model's main params '''
        ## print a loogBookFile-.
        ## print(dir_logs)
        log_file=dir_logs+'/'+ 'log.p'; file_log=open(log_file,'wb') 
        ## print(log_file, type(log_file), sep='\n')
        ## '/gpfswork/rech/irr/uhe87yl/carazof/scratch/fernando/resDL_2_SS/log.p'

        ## dictionary2print-.
        log_dict={'learning_rate': self.learning_rate,
                  'batch_size_train': bst,
                  'batch_size_val': bsv,
                  'num_epochs': epochs,
                  'layers_list': self.layers_data,
                  'optimizer': optimizer, # I don't save self.optimizer 2don't save torc.optim ..
                  'weight_decay': self.wd,
                  'momentum': self.momentum,
                  'loss': loss, # I don't save self.loss 2don't save torc.optim ..
                  'dataset_file': ds_file,
                  'folder_out': dir_res,
                  }
        ## with open(log_file,'wb') as fn: pickle.dump(log_dict,file_log); fn.close()
        pickle.dump(log_dict,file_log)
        file_log.close()
# - =======================================================================END79

if __name__ =='__main__': pass