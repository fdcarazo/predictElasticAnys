#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## class to laod and process dataset/s-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Tue Feb  6 10:42:38 CET 2024 -.
## last_modify (Fr): Tue Feb  6 11:34:31 CET 2024-.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
from os.path import join as join
import glob
import pandas as pd

## main class-.
## 2BeMod: -.
class Dataset():
    '''
         A class to load and process dataset used to test-.
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
    def __init__(self, root_ds:str, ds_file:str):
        ''' constructor '''
        ## <class 'list'> -DF used to TEST the model-.
        self.df_name= glob.glob(join(str(root_ds), ds_file))

    def set_cols(self,dim:str) -> list:
        cols=list()
        if dim=='2D' or dim=='2d':
            '''
            cols=['irun','vgrad11', 'vgrad22', 'vgrad12', 'vgrad21','c11_in',
                  'c22_in', 'c33_in', 'c44_in', 'c55_in', 'c66_in', 'c56_in',
                  'c46_in', 'c36_in', 'c26_in', 'c16_in', 'c45_in', 'c35_in', 
                  'c25_in', 'c15_in', 'c34_in', 'c24_in', 'c14_in', 'c23_in',
                  'c13_in', 'c12_in', 'c11_out', 'c22_out', 'c33_out', 'c44_out',
                  'c55_out', 'c66_out', 'c56_out', 'c46_out', 'c36_out', 'c26_out', 
                  'c16_out', 'c45_out', 'c35_out', 'c25_out', 'c15_out', 'c34_out', 
                  'c24_out', 'c14_out', 'c23_out', 'c13_out', 'c12_out','strain']
            '''
            '''
            cols=['irun','vgrad11', 'vgrad22', 'vgrad12', 'vgrad21','c11_in',
                  'c22_in', 'c33_in', 'c44_in', 'c55_in', 'c66_in',
                  'c26_in', 'c16_in', 'c45_in', 'c23_in',
                  'c13_in', 'c12_in', 'c11_out', 'c22_out', 'c33_out', 'c44_out',
                  'c55_out', 'c66_out', 'c56_out', 'c46_out', 'c36_out', 'c26_out',
                  'c16_out', 'c45_out', 'c35_out', 'c25_out', 'c15_out', 'c34_out',
                  'c24_out', 'c14_out', 'c23_out', 'c13_out', 'c12_out','strain']
            '''
            cols=['irun','vgrad11', 'vgrad12', 'vgrad21','c11_in',
                  'c22_in', 'c33_in', 'c44_in', 'c55_in', 'c66_in', 'c56_in',
                  'c46_in', 'c36_in', 'c26_in', 'c16_in', 'c45_in', 'c35_in', 
                  'c25_in', 'c15_in', 'c34_in', 'c24_in', 'c14_in', 'c23_in',
                  'c13_in', 'c12_in', 'c11_out', 'c22_out', 'c33_out', 'c44_out',
                  'c55_out', 'c66_out', 'c56_out', 'c46_out', 'c36_out', 'c26_out', 
                  'c16_out', 'c45_out', 'c35_out', 'c25_out', 'c15_out', 'c34_out', 
                  'c24_out', 'c14_out', 'c23_out', 'c13_out', 'c12_out','strain']
        elif dim=='3D' or dim=='3d':
            cols=['irun','vgrad11', 'vgrad22', 'vgrad33', 'vgrad23', 'vgrad13',
                  'vgrad12', 'vgrad32', 'vgrad31', 'vgrad21', 'c11_in', 'c22_in',
                  'c33_in', 'c44_in', 'c55_in', 'c66_in', 'c56_in', 'c46_in',
                  'c36_in', 'c26_in', 'c16_in', 'c45_in', 'c35_in', 'c25_in',
                  'c15_in', 'c34_in', 'c24_in', 'c14_in', 'c23_in', 'c13_in',
                  'c12_in', 'c11_out', 'c22_out', 'c33_out', 'c44_out', 'c55_out',
                  'c66_out', 'c56_out', 'c46_out', 'c36_out', 'c26_out', 'c16_out',
                  'c45_out', 'c35_out', 'c25_out', 'c15_out', 'c34_out', 'c24_out',
                  'c14_out', 'c23_out', 'c13_out', 'c12_out','strain']
        return cols

    def load_ds(self, dim:str):
        cols=self.set_cols(dim)

        ## df= pd.read_csv(self.df_name[0], index_col=0, low_memory=False, usecols=cols)
        df=pd.read_csv(self.df_name[0], low_memory=False, usecols=cols)
        
        ## get features and targets variables names-.
        target_names, feature_names=[col for col in df.columns if '_out' in col],\
            [col for col in df.columns if '_in' in col or 'vgrad' in col]
        return df, feature_names, target_names

if __name__=='__main__':
    dataset_path='/Users/Fernando/temp/' # in a real life it's read in Config class-.
    dataset_file='db3-SSb.csv' # in a real life it's read in Config class-.
    ds_obj=Dataset(dataset_path,dataset_file) # in a real life it's read in Config class-.
    ## print(ds_obj)
    ## columns=ds_obj.set_cols('2D') # in a real life it's read in Config class-.
    ## print(columns)
    df_main, feat_var, tar_var=ds_obj.load_ds('3D') # in a real life it's read in Config class-.
    print('{}{}{}'.format('\n'*2,df_main,'\n'*2))
    print('{0}{1}{2}{0}'.format('\n'*2,feat_var, len(feat_var)))
    print('{0}{1}{2}{0}'.format('\n'*2,tar_var, len(tar_var)))
    
