#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: NN (MAML -Meta Agnostic Machine Learning)) from scratch used to 
##         train the model in UM's or JeanZay's Cluster-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## start date (Fr): Wed Mar  6 11:50:18 CET 2024 -.
## last modify (Fr): -.
##
## ====================================================================== INI79

## ====================================================================== INI79
## 1- include packages, modules, variables, etc.-.
import torch
import torch.nn as nn
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
class MamlNet(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 layers_data: list,
                 device='cpu',
                 learning_rate=1.0e-2,
                 optimizer=optim.SGD,
                 loss=nn.L1Loss(),
                 weight_decay=1.0e-2,
                 momentum=0.99,
                 optimizer_inner=optim.SGD, # inner loop MAML from scratch-.
                 n_o_u=100, # numOfUpdate in Outer loop in MAML from scratch-.
                 n_i_u=1, # numOfUpdate in Inner loop in MAML from scratch-.
                 lr_o=1.0e-3, # learning rate in Outer loop in MAML from scratch-.
                 lr_i=1.0e-3 # learning rate in Inner loop in MAML from scratch-.
                 ):        
        super(MamlNet,self).__init__()
    
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
        self.optimizer=optimizer(params= self.parameters(), lr=self.learning_rate)
        self.loss=loss # nn.MSELoss()
        self.wd=weight_decay
        self.momentum=momentum
        ## MAML form scratch-.
        self.n_o_u=n_o_u # numOfUpdate in Outer loop in MAML from scratch-.
        self.n_i_u=n_i_u, # numOfUpdate in Inner loop in MAML from scratch-.
        self.lr_o=lr_o # learning rate in Outer loop in MAML from scratch-.
        self.lr_i=lr_i # learning rate in Inner loop in MAML from scratch-.
        self.optimizer_inner=optimizer_inner(params=self.parameters(), lr=self.lr_i)
        
    def forward(self,inputs,params=None):
        for layer in self.layers: inputs=layer(inputs)
        return inputs

    def save_params_model(self,dir_logs,bst,bsv,epochs,optimizer,loss,ds_file,dir_res,optim_inner):
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
                  'number_outer_update': self.n_o_u, # numOfUpdate in Outer loop in MAML from scratch-.
                  'number_inner_update': self.n_i_u, # numOfUpdate in Inner loop in MAML from scratch-.
                  'learning_rate_outer': self.lr_o, # learning rate in Outer loop in MAML from scratch-.
                  'learning_rate_inner': self.lr_i, # learning rate in Inner loop in MAML from scratch-.
                  'optimizer_inner': optim_inner
                  }
        ## with open(log_file,'wb') as fn: pickle.dump(log_dict,file_log); fn.close()
        pickle.dump(log_dict,file_log)
        file_log.close()

# - =======================================================================END79

if __name__ =='__main__': pass
