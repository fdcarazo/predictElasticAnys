#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: NN (BNN) used to train the model in UM's or JeanZay's Cluster-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## start date (Fr): Mon Mar  4 17:50:44 CET 2024 -.
## last modify (Fr): Tue Mar  5 16:56:00 CET 2024 -.
##
## ====================================================================== INI79

## ====================================================================== INI79
## 1- include packages, modules, variables, etc.-.
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pickle

import torchbnn as bnn

## 2don't have problems between float datatype of torch and bnnNN
## if torch.is_tensor(xx) else torc.tensor(xx,dtype=float) (float64) and float
## of NN (float32)-.
torch.set_default_dtype(torch.float64)
## torch.set_default_dtype(torch.float32)
## - ======================================================================END79

## - ======================================================================INI79
## Bayesian NN using torchbnn package-.
class BayesianNet(nn.Module):
    def __init__(self,
                 inp:int,
                 out:int,
                 loss=torch.nn.L1Loss(),
                 optimizer=optim.SGD,
                 kl_loss=bnn.BKLLoss(),
                 device='cpu',
                 w_d=0.0, #1.0e-2,
                 lr=1.0e-2,
                 kl_weight=1.0e-1
                 ): # 1-100-1-.
        ## NN-topology or architecture-.
        super(BayesianNet, self).__init__()

        self.inp=inp; self.out=out
        self.device=device
        self.w_d=w_d
        self.l_r=lr
        self.kl_weight=kl_weight
        
        self.hid1=bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                  in_features=self.inp, out_features=2048
                                  )
        self.ouput=bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                  in_features=2048, out_features=self.out
                                  )
        self.relu=nn.ReLU()

        ## NN-loss function, optimizer, loss and KL-loss-.
        ## NOTE: optimizer and losses shoulb came after define topology or 
        ## architecture of NN-.
        self.optimizer=optimizer(params=self.parameters(), lr=self.l_r, weight_decay=self.w_d)
        self.loss=loss
        self.kl_loss=kl_loss
        
    def forward(self,inputs):
        z=self.relu(self.hid1(inputs))
        return self.ouput(z)

    def save_params_model(self,dir_logs,bst,bsv,epochs,optimizer,loss,ds_file,dir_res,kl_l):
        ''' method to save the DL model's main params '''
        ## print a loogBookFile-.
        ## print(dir_logs)
        log_file=dir_logs+'/'+ 'log.p'; file_log=open(log_file,'wb') 
        ## print(log_file, type(log_file), sep='\n')
        ## '/gpfswork/rech/irr/uhe87yl/carazof/scratch/fernando/resDL_2_SS/log.p'

        ## dictionary2print-.
        log_dict={'learning_rate': self.l_r,
                  'batch_size_train': bst,
                  'batch_size_val': bsv,
                  'num_epochs': epochs,
                  ## 'layers_list': self.layers_data,
                  'optimizer': optimizer, # I don't save self.optimizer 2don't save torc.optim ..
                  'weight_decay': self.w_d,
                  ##'momentum': self.momentum,
                  'loss': loss, # I don't save self.loss 2don't save torc.optim ..
                  'dataset_file': ds_file,
                  'folder_out': dir_res,
                  'kl_weight': self.kl_weight,
                  'kl_loss': kl_l

                  }
        ## with open(log_file,'wb') as fn: pickle.dump(log_dict,file_log); fn.close()
        pickle.dump(log_dict,file_log)
        file_log.close()

## - ======================================================================END79