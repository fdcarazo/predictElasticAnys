#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: NN (FFNN) used to train the model in UM's or JeanZay's Cluster-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## start date (Fr): Sun Mar  3 22:48:52 CET 2024-.
## last modify (Fr): Tue Mar  5 12:14:35 CET 2024-.
##
## ====================================================================== INI79

## ====================================================================== INI79
## 1- include packages, modules, variables, etc.-.
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pickle
## - ======================================================================END79

## - ======================================================================INI79
## DL -- FFNN class-.
class MLP(nn.Module):
    def __init__(self, input_size,
                 output_size,
                 layers_data: list,
                 device='cpu',
                 learning_rate=1.0e-2,
                 optimizer=optim.SGD,
                 loss=nn.L1Loss(),
                 weight_decay=1.0e-2,
                 momentum=0.99
                 ):
        super().__init__()

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
        ## self.criterion=self.set_criterion(loss) # nn.MSELoss()
        self.wd=weight_decay
        ## self.optimizer=self.set_optimizer('Adam',lr=self.learning_rate,weight_decay=self.wd)
        self.momentum=momentum

    def forward(self,inputs):
        ''' forward '''
        for layer in self.layers: inputs=layer(inputs)
        return inputs

    def save_params_model(self,dir_logs,bst,bsv,epochs,optimizer,loss,ds_file,dir_res,num_est):
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
                  'num_estimators': num_est
                  }
        ## with open(log_file,'wb') as fn: pickle.dump(log_dict,file_log); fn.close()
        pickle.dump(log_dict,file_log)
        file_log.close()
# - =======================================================================END79

if __name__=='__main__': pass
