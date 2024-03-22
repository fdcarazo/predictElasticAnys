#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## class to sdefine the folder to save figs-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Tue Feb  6 15:59:45 CET 2024-.
## last_modify (Fr): Tue Feb  6 16:04:33 CET 2024-.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
from os.path import exists as ope
from os import makedirs as om
import sys

sys.path.append('/Users/Fernando/scratch/elasAnys/2testModels/')
from utils.gen_tools import get_args  as ga
from utils import gen_tools as gt

## main class-.
## 2BeMod: -.
class SaveFigs():
    '''
         A class define the folder to save figs-.
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
    def __init__(self,save:bool,ds_file:str,dir_save:str,root:str):
        self.save=save
        self.ds_file=ds_file
        self.dir_save=dir_save
        self.root=root
        
        if self.save:
            self.folder_save= gt.folder_to_save_figs(self.ds_file, self.dir_save, self.root)
            ## check if folder exist, if not cerate this-.
        is_folder_exist=ope(self.folder_save)
        if not is_folder_exist:
            om(self.folder_save)
            print('The folder ===< {0}{1}{2} >=== was created.'.format('\t', self.dir_save, '\t'))
        else:
            print('The folder ===< {0}{1}{2} >=== exists.'.format('\t', self.dir_save, '\t'))
    ## ======================================================================= END79

if __name__=='__main__':
    ''' main function '''
    sys.path.append('/Users/Fernando/scratch/elasAnys/2testModels/')
    from utils.gen_tools import get_args  as ga
    from utils import gen_tools as gt
    ##
    dataset_path='/Users/Fernando/temp/' # in a real life it's read in Config class-.
    save_figs=True
    ds_file='db3-SSb.csv' # /Users/Fernando/temp/
    root_save_figs='/Users/Fernando/temp/'
    root='/Users/Fernando/temp/'
    
    ## create object
    sf=SaveFigs(save_figs, ds_file, root_save_figs, root)
    ## print(type(mlmodel)) ## MLP.MPL
    # in a real life it's read in Config class-.
    print('{0}{1}{0}'.format('\n'*2,sf,'\n'*2))
    print('{0}{1}{0}'.format('\n'*2,dir(sf),'\n'*2))
    print('{0}{1}{0}'.format('\n'*2,sf.save_figs,'\n'*2))
    print('{0}{1}{0}'.format('\n'*2,sf.folder_figs,'\n'*2))

