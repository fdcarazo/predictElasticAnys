#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: used in main & process_plot scripts-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
## last_modify: Wed Jan 31 10:40:23 CET 2024-.
##

## ======================================================================= INI79
## 1- include packages, modules, variables, etc.-.
from pathlib import Path
import yaml
import pandas as pd
import torch as torch
## print(dir()) # to see the names in the local namespace-.
## ======================================================================= END79

## ======================================================================= INI79
## 2- Classes, Functions, Methods, etc. definitions-.

## ======================================================================= INI79
## 2-1- remove columns with all values are equal to zero-.
def remove_cols_zeros(df):
    '''
    function remove_cols_zeros: remove columns with all values are null/Nan-.
    Variables:
    1- df: df to remove features which values are zero-.
    2- Output:
    df: df with zeros columns removed-.
    n_cols_rem: numbers of columns removed-.
    '''
    n_cols_rem=0 # by default =0-.
    # find the zeros/nulls/NaNs-.
    zeroes= (df==0.0) & (df.applymap(type)==float)
    # find columns with only zeroes-.
    cols= zeroes.all()[zeroes.all()].index.to_list() # list with the indices of the columns to delete-.
    n_cols_rem= len(cols) # numbers of the columns to delete-.
    # drop these columns-.
    df= df.drop(cols, axis=1) # remove columns with all values equal to zero-.

    return df, n_cols_rem
## ======================================================================= END79

## ======================================================================= INI79
## 2-2- print the differences between two lists -.
def print_remove_cols(df_in, df_out):
    '''
    function print_remove_cols: print the differences between two lists. In this case 
                                is used to print the name of removed columns in a
                                pandasDF -.

    df_in: list with a columns/features names of original pandasDF-.
    df_out: list with a columns/features names of modified pandasDF-.
    '''
    print(set(list(df_out)) - set(list(df_in)))
## ======================================================================= END79

## ======================================================================= INI79    
## 2-3- 2set the name of folder 2 save figures-.
def folder_to_save_figs(ds_file_name: str, root_save_figs: str, r) -> str:
    '''
    function folder_to_save_figs: set the name of the folders in which I 
                                  will save the figures-.
    Variables:
    ds_file_name: name of the file to test-.
    r (acronymous of 'root'): root folder (set in exec_env fucntion)-.
    Output:
    f_f (acronymous of 'figure_folders'): absolute PATH in which I will save
                                          the figures-.
    '''

    fold_name= str.split(ds_file_name, '.')[0]+'_figs'
    f_f= root_save_figs+ fold_name

    return f_f
## ======================================================================= END79

# - =======================================================================INI79
## 2-4- calculate elastic anisotropy coefficients according to et al.-.
rd={'type':float, 'units':'adimensional',
    'docstring':'anisotropy coefficients according to: et al.-'
    }
def calcula_elas_anys_coef(c11, c12, c13, c23, c33, c44, c55, c66)->rd:
    '''
    function to calculate anisotropy coefficients
    according to: et al.-.
    '''
    epsilon=( 1./8.*(c11+c33)- 1./4.*c13+ 1./2.*c55 ) / ( 1./2.*(c44+c66))
    eta= (1./2.* (c12+c23))/ (3./8.*(c11+c33)+ 1./4.*c13+ 1./2.*c55- (c44+c66))
    return epsilon, eta

# - =======================================================================INI79
## 2-6- get the arguments from the config file-.
def get_args(path: Path):
    '''
    function get_args: get the arguments from the config file-.
    '''
    with open(path, 'r') as f: config= yaml.safe_load(f)
    return config
# - =======================================================================END79
