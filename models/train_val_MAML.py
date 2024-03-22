#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: 2train and test (with validation DS) any FFNN or Maml-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## start date (Fr): Mon Mar  4 12:39:33 CET 2024-.
## last modify (Fr): Tue Mar  5 13:21:14 CET 2024-.
##
## ====================================================================== INI79

## ====================================================================== INI79
## 1- include packages, modules, variables, etc.-.
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm as tqdm
import time as time
from copy import deepcopy

from MLP import MLP
## - ======================================================================END79
## - ======================================================================INI79
## 2- To train and test models-.
class TrainPredict(MLP):
    def __init__(self,model,train_loader,val_loader,inp,out,layers,device,lr,
                 optim,wd,mom,optim_inner,nou,niu,lro,lri): # some params are needed to inherit MLP class (for constructor)-.)
        ''' class constructor '''
        super().__init__(inp,out,layers) # ,layers,device,lr,optim,wd,mom
        
        self.model=model
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.device=device
        self.nou=nou # number_outter_update-.
        self.niu=niu # number_inner_update-.
        self.lro=lro # learning_rate_outter-.
        self.lri=lri # learning_rate_inner-.
        
    def train(self,epochs=1, val=False):
        ''' FeedForwardNeuralNetwork -FFNN- trainer '''
        loss_dir={'train_loss': [],
                  'val_loss': []
                  }
        train_time=0.0
        n_train=len(self.train_loader)
        ## train
        ## -> 2apply dropout/normalization (only in train) and 2Calc and Save the
        ##    grad of params W.R.T. loss function which T.B.U. to update model 
        ##    params (weights and biases)-.
        self.model.train()

        epochs=tqdm.tqdm(range(epochs), desc='Epochs')
        start_time=time.time()
        for iepoch in epochs:
            ##########for iepoch in range(epochs):
            ##########with tqdm.tqdm(total=n_train,position=0) as pbar_train:
            ##########   pbar_train.set_description('Epoch -- {0} -- / '+'({1})'+ ' - train- {2}'.
            ##########                              format(epoch+1,'epoch','\n')
            ##########                               )
            ##########    ## pbar_train.set_postfix(avg_loss='0.0')
            loss_dir['train_loss'].append(self.fit_regression_ffnn(self.train_loader,
                                                                   ## pbar_train,
                                                                   True))
            ## val loop-.
            if val:
                n_val=len(self.val_loader)
                ## -> if exists (layers), don't applied dropout, batchNormalization,
                ## and don't registers gradients among other things-.
                self.model.eval() 
                ## with tqdm(total=n_val, position=0) as pbar_val:
                ## pbar_val.set_description('Epoch -- {0} -- / '+'({1})'+ ' - val-.'.
                ##                             format(epoch+1,'epoch')
                ##                             )
                ## pbar_val.set_postfix(avg_loss='0.0')
                loss_dir['val_loss'].append(self.fit_regression_ffnn(self.val_loader,
                                                                     ## pbar_val,
                                                                     False))
        train_time=time.time()- start_time # per epoch-.
        return train_time, loss_dir
    
    def fit_regression_ffnn(self,dataloader,train=True):
        ''' train and/or validation using batches '''
        running_loss=0.0
        for idl, (X,Y) in enumerate(dataloader,0):
            model_copy=deepcopy(self.model)
            ## X,Y=map(lambda x: x.to('cpu'),data) # cpu or gpu-.
            X,Y=X.to(self.device),Y.to(self.device) ## map(lambda x: x.to('cpu'),data) # cpu or gpu-.
            ## inner loop-.
            if train: 
                for _ in range(self.niu):
                    y_pred=model_copy(X)
                    loss=model_copy.loss(y_pred,Y)
                    model_copy.optimizer_inner.zero_grad()
                    loss.backward()
                    model_copy.optimizer_inner.step()
                    
                    ## print(X.get_device(), Y.get_device(), sep='\n'); input(11)
                    ########## self.model.optimizer.zero_grad() # re-start gradient-.
                    model_copy.optimizer.zero_grad() # re-start gradient-.
            ########## pred=self.model_copy(X) # forward pass -.
            pred=model_copy(X) # forward pass -.
            ########## loss=self.model.loss(pred,Y) # evaluate prediction-.
            loss=model_copy.loss(pred,Y) # evaluate prediction-.
            if train:
                ## apply loss back propropagation-.
                loss.backward() # backpropagation-.
                ########## self.model.optimizer.step() # update parameters (weights and biases))-.
                model_copy.optimizer.step() # update parameters (weights and biases))-.
            self.model=model_copy
            running_loss+=loss.item()
        avg_loss=running_loss/len(X)
            ########## pbar.set_postfix(avg_loss='{:.4f}'.format(avg_loss))
            ## pbar.update(Y.shape[0]) # ?-.
        return avg_loss

    def fit_regression_ffnn_1(self,dataloader,train=True):
        ''' train and/or validation using batches '''
        running_loss=0.0
        for idl, (X,Y) in enumerate(dataloader,0):
            ## X,Y=map(lambda x: x.to('cpu'),data) # cpu or gpu-.
            X,Y=X.to(self.device),Y.to(self.device) ## map(lambda x: x.to('cpu'),data) # cpu or gpu-.
            ## inner loop-.
            if train: 
                for _ in range(self.niu):
                    y_pred=self.model(X)
                    loss=self.model.loss(y_pred,Y)
                    self.model.optimizer_inner.zero_grad()
                    loss.backward()
                    self.model.optimizer_inner.step()
                    
                    ## print(X.get_device(), Y.get_device(), sep='\n'); input(11)
                    ########## self.model.optimizer.zero_grad() # re-start gradient-.
                    self.model.optimizer.zero_grad() # re-start gradient-.
            ########## pred=self.self.model(X) # forward pass -.
            pred=self.model(X) # forward pass -.
            ########## loss=self.model.loss(pred,Y) # evaluate prediction-.
            loss=self.model.loss(pred,Y) # evaluate prediction-.
            if train:
                ## apply loss back propropagation-.
                loss.backward() # backpropagation-.
                ########## self.model.optimizer.step() # update parameters (weights and biases))-.
                self.model.optimizer.step() # update parameters (weights and biases))-.
            running_loss+=loss.item()
        avg_loss=running_loss/len(X)
            ########## pbar.set_postfix(avg_loss='{:.4f}'.format(avg_loss))
            ## pbar.update(Y.shape[0]) # ?-.
        return avg_loss

## - ======================================================================END79
