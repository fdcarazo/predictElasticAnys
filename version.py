#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: list and save in 'modules_versions.txt' the packages
##         names and versions used in the project-.
## NOTE: some packages don't have __version__ or version methods-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## last_modify: Wed Jan 31 13:35:24 CET 2024-.
##

## ======================================================================= INI79
## 1- from 2plotresults/utils-.
import pathlib as pl
import yaml as yaml
import pandas as pd
import torch as torch

## 2- from 2plotresults/src-.
## import pandas as pd # loaded in 1- -.
import matplotlib as mpl
import seaborn as sns
import numpy as np
import time, pickle, os, psutil, glob, typing, warnings
## import torch # loaded in 1- -.

## 3- from models-.
## import pandas as pd # loaded in 1-
## import torch # loaded in 1- -.
import sys

import importlib # importlib.reload(module) # 2reload a module-.
from importlib.metadata import version # as an alternative to __version__

class mod_versions():
    def __init__(self):
        ## self.pathlib_ver=pathlib.__version__
        self.yaml_ver= yaml.__version__
        self.pandas_ver= pd.__version__
        self.torch_ver= torch.__version__
        self.matplotlib_ver=mpl.__version__
        self.seaborn_ver=sns.__version__
        self.numpy_ver=np.__version__
        self.np_ver=version('numpy')
        ## self.time_ver=time.__version__
        ## self.time_ver=version('time')
        ## self.pickle_ver=pickle.__version__
        ## self.os_ver=os.__version__
        self.psutil_ver=psutil.__version__
        ## self.glob_ver=glob.__version__
        ## self.typing_ver=typing.__version__
        ## self.warnings_ver=warnings.__version__
        self.sys_ver=sys.version
        ## all versions-.
        '''
        self.all_mod_versions=[pathlib.__version__, yaml.__version__, pandas.__version__,
                               torch.__version__, matplotlib.__version__, seaborn.__version__,
                               numpy.__version__, time.__version__, pickle.__version__,
                               os.__version__, psutil.__version__, glob.__version__,
                               typing.__version__, warnings.__version__, sys.__version__
                               ]
        '''
    
    def get_attribute(self):
        ''' prints all attribute of object '''
        for i in (vars(self)):
            print('{0:10}: {1}'.format(i, vars(self)[i]))
        '''
        ## name & complete PATH of modules loaded bt me-.
        for name, module in sorted(sys.modules.items()): 
            if hasattr(module, '__version__'): 
                print('{} ||  {}'.format(name, module.__version__))
        '''
        '''
        ## only modules loaded by me
        modulenames = set(sys.modules) & set(globals()) # only name as dict
        allmodules = [sys.modules[name] for name in modulenames] # name and complete PATH-.
        print(modulenames); print(allmodules)
        '''
        
    def open_save_modules(self):
        name_file='modules_version.txt'
        try:
            f2r=open(name_file) # os.path.isfile(path) os.path.exists(path)
            print("{0}File{1} exist. It'll deleted{0}".format('\n',name_file,))
        except FileNotFoundError as e:
            print("{0}File{1} doesn't exist. It'll be created{0}".format('\n',name_file,))
            
        with open(name_file, 'w') as f:
            for i in (vars(self)): f.write('{0:10}: {1}{2}'.format(i, vars(self)[i], '\n'))
            f.close()

            

            
