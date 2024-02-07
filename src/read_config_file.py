#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## class to read config file-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Tue Feb  6 10:22:52 CET 2024 -.
## last_modify (Fr): Wed Feb  7 12:21:47 CET 2024-.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
from os.path import dirname as drn, realpath as rp
from typing import Dict
import yaml

## from ..utils.gen_tools import get_args  as ga
## from ..utils import gen_tools as gt

## main class-.
## 2BeMod: set attributes as private and use getter and setter methods,
##         also delete object after it is used-.
class Config():
    '''
        Aclass to load config file-.
    ...
    Attributes (only an example, 2BeCompleted-.)
    ----------
    name : str
        first name of the person
    Methods  (only an example, 2BeCompleted-.)
    -------
    info(additional=""):
        Prints the person's name and age.
    '''
    
    def __init__(self, cfg: Dict):
        ''' constructor '''
        self.config= cfg # in
        ## out-.
        self.save_figs=self.config['gen_options']['save_figs'] # 2save or not the figures-.
        self.dir_save_figs= self.config['gen_options']['dir_save_figs'] # 2set path save figures-.
        self.dim=self.config['gen_options']['dim']
        
        ## 2-2- Paths-.
        ## self.currentdir=drn(rp(__file__))
        ## self.config_file_path='/Users/Fernando/scratch/elasAnys/2testModels/config_file.yaml'
        ## self.root=drn(self.currentdir)
        ## self.root_ds=self.config['dataset']['ds_path']
        
        ## 2-2- Dataset names-.
        ## datasets (used to train and test)-.
        self.ds_path=self.config['dataset']['path']
        self.ds_file=self.config['dataset']['name']
        
        ## folder 2 load the ML/DL model-.ou
        ## machine learning file name-.
        self.root_mod_sca=self.config['model']['path'] ## ml_path= config['mlmodel']['ml_path']
        self.mfn=self.config['model']['name']
        
        ## scaler/standarizer names-.
        self.sca_feat=self.config['scaler']['feat'] # used to scale/standarize a FEATURES of the DF-.
        self.sca_targ=self.config['scaler']['targ'] # used to scale/standarize a TARGETS of the DF-.

if __name__=='__main__':
    config_file_path='/Users/Fernando/scratch/elasAnys/2testModels/config_file.yaml'
    with open(config_file_path, 'r') as f: config= yaml.safe_load(f)
    cfg_obj=Config(config)
    print(cfg_obj.__dir__(), dir(cfg_obj), sep='\n'*2)
    print('{}{}'.format('\n'*3,cfg_obj.__dict__))
    print(cfg_obj.ensemble_dl, cfg_obj.mfn, sep='\n')
else:
    print('{0} imported as Module'.format(__file__.split('/')[-1]))

