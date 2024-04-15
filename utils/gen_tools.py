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
import os
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from numpy import linalg as LA
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
    zeroes=(df==0.0) & (df.applymap(type)==float)
    # find columns with only zeroes-.
    cols=zeroes.all()[zeroes.all()].index.to_list() # list with the indices of the columns to delete-.
    n_cols_rem=len(cols) # numbers of the columns to delete-.
    # drop these columns-.
    df=df.drop(cols, axis=1) # remove columns with all values equal to zero-.

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
def calcula_elas_anys_coef(c11, c13, c22, c33, c44, c55, c66)->rd:
    '''
    function to calculate anisotropy coefficients
    according to: et al.-.
    '''
    epsilon=( 1./8.*(c11+c33)- 1./4.*c13+ 1./2.*c55 ) / ( 1./2.*(c44+c66))
    ## eta= (1./2.* (c12+c23))/ (3./8.*(c11+c33)+ 1./4.*c13+ 1./2.*c55- (c44+c66))
    phi= (c22)/ (3./8.*(c11+c33)+ 1./4.*c13+ 1./2.*c55)
    return epsilon, phi

# - =======================================================================INI79
## 2-5- get the arguments from the config file-.
def get_args(path: Path):
    '''
    function get_args: get the arguments from the config file-.
    '''
    with open(path, 'r') as f: config= yaml.safe_load(f)
    return config
# - =======================================================================END79

# - =======================================================================INI79
## 2-6- to interpolate a straight line to a cloud of points-.
def adjust_line(x,y):
    ''' coefficients of stringht line-. '''
    pearR=np.corrcoef(x.to_numpy(dtype=float), y)[1,0]
    A=np.vstack([x.to_numpy(dtype=float),np.ones(len(x))]).T
    m,c=np.linalg.lstsq(A,y)[0]
    return m,c,pearR
# - =======================================================================END79

# - =======================================================================INI79
## 2-7- calculate different regression metrics between two pandas.DataFrame
def eps_eta_rmse(y_true,y_pred):
    ## metrics calculations-.
    mse=mean_squared_error(y_true,y_pred)
    r2=r2_score(y_true,y_pred)
    rmse=np.sqrt(mean_squared_error(y_true,y_pred))
    '''
        print('Predicted case {0:>2}{1}{2:<6}{3:>2}{4:>10.6f}{5}{6:<6}{7:>2}{8:>10.6f}{9}{10:<6}{11:>2}{12:>10.6f}{13}'.
              format(case,'\n','MSE','=',mse,'\n','R2','=',r2,'\n','RMSE','=',rmse,'\n'))
        print('++'*30)
        print('EPSILON_MSE (TEO-PRED) = {0:>12.9f}'.
              format(mean_squared_error(eval(f'eps_teo_{iruned[ic]}'),eval(f'eps_pred_{iruned[ic]}'))))
        print('EPSILON_RMSE (TEO-PRED) = {0:>12.9f}'.
              format(np.mean(np.sqrt(np.square(diff_pred_true_eps.astype(float))))))
        print('ETA_MSE (TEO-PRED) = {0:>12.9f}'.
              format(mean_squared_error(eval(f'eta_teo_{iruned[ic]}'),eval(f'eta_pred_{iruned[ic]}'))))
        print('ETA_RMSE (TEO-PRED) = {0:>12.9f}'.
              format(np.mean(np.sqrt(np.square(diff_pred_true_eta.astype(float))))))
        print('++'*30)
    '''
    return rmse

# - =======================================================================INI79
## 2-8- calculate different regression metrics between two pandas.DataFrame
def print_metrics(df1,df2)->int:
    '''general function to calculate emtrics between two daatsets-.'''
    ## print metrics-.
    print('Predicted case (IRUN) {0:>2}{1}{2:<6}{3:>2}{4:>10.6f}{5}'+
          '{6:<6}{7:>2}{8:>10.6f}{9}{10:<6}{11:>2}{12:>10.6f}{13}'.
          format(case,
                 '\n','MSE','=',np.round(mean_squared_error(y_pred, y_test), 6),
                 '\n','R2','=',np.round(r2_score(y_pred, y_test), 6),
                 '\n','RMSE','=',np.round(np.sqrt(mean_squared_error(y_pred, y_test)), 6),
                 '\n'
                 ))
    return 0
# - =======================================================================END79

## - =======================================================================INI79
## 2-9- function to calculate the distance between tensors-.
def calcula_dist_bet_tensors(tens1, tens2):
    '''
    ==========
    tens1, tens2: tensors to calculate the ditance-.
    OUTPUT:
    ======
    euclideanDistance between two tensores-.
    '''
    ## return np.sqrt(np.sum(np.square(tens1 - tens2)))
    return LA.norm((tens1-tens2), axis=1)
## ======================================================================= END79

## - =======================================================================INI79
## 2-10- write a complete $C^{out}_{ij}$ from one VPSC/IRUN to read and plot in
##      MTEX
def write_C_for_mtex(df_vpsc,df_pred_rec,df_pred,dir_save:str,case_name:str):    

    '''
    Arguments
    ==========
    df: df with a complete IRUN/VPSC case-.
    [11,22,33,44,55,66,56,46,36,26,16,45,35,25,15,34,24,14,23,13,12]
    
    OUTPUT:
    ======
    6x6 matrix representation of a df in .txt format to be abble to
    read in MTEX-.
    '''

    if len(df_vpsc)==1:
        df_vpsc=df_vpsc[0]; df_pred=df_pred[0]; df_pred_rec=df_pred_rec[0]
    else:
        idx=np.random.randint(0,len(df_vpsc))
        df_vpsc=df_vpsc[idx];df_pred=df_pred[idx];df_pred_rec=df_pred_rec[idx]

    df_dict={'C_true': df_vpsc, 'C_pred': df_pred, 'C_pred_rec': df_pred_rec}

    for ic,(key,df) in enumerate(df_dict.items()):
        c11=df[['c11_out']];c22=df[['c22_out']];c33=df[['c33_out']];c44=df[['c44_out']];c55=df[['c55_out']]
        c66=df[['c66_out']];c56=df[['c56_out']];c46=df[['c46_out']];c36=df[['c36_out']];c26=df[['c26_out']]
        c16=df[['c16_out']];c45=df[['c45_out']];c35=df[['c35_out']];c25=df[['c25_out']];c15=df[['c15_out']]
        c34=df[['c34_out']];c24=df[['c24_out']];c14=df[['c14_out']];c23=df[['c23_out']];c13=df[['c13_out']]
        c12=df[['c12_out']]
        if ic==0: strain=df[['strain']]
        
        ## print(strain['strain'].iloc[0])
        ## print(strain.loc[il,'strain'])
        f_name=dir_save+key+'.txt'

        if os.path.exists(f_name): os.remove(f_name)
        f=open(f_name,'a')
        f.write('{0}{1}'.format(case_name,'\n'*2))
        ## print(enumerate(range(len(df)))), input(55)
        for il,_ in enumerate(range(len(df))):
            f.write('{0}{1}{2}{3}{4}{2}'.
                    format('matrix',str(il),'\n','strain=',str(strain['strain'].iloc[il])))
            f.write('{0}{1}'.format('BEGIN','\n'))
            # row 1-.
            f.write('{0}{1}{2}{1}{3}{1}{4}{1}{5}{1}{6}{7}'.
                    format(c11['c11_out'].iloc[il],'\t',c12['c12_out'].iloc[il],
                           c13['c13_out'].iloc[il],c14['c14_out'].iloc[il],
                           c15['c15_out'].iloc[il],c16['c16_out'].iloc[il],'\n'))
            # row 2-.
            f.write('{0}{1}{2}{1}{3}{1}{4}{1}{5}{1}{6}{7}'.
                    format(c12['c12_out'].iloc[il],'\t',c22['c22_out'].iloc[il],
                           c23['c23_out'].iloc[il],c24['c24_out'].iloc[il],
                           c25['c25_out'].iloc[il],c26['c26_out'].iloc[il],'\n'))
            # row 3-.
            f.write('{0}{1}{2}{1}{3}{1}{4}{1}{5}{1}{6}{7}'.
                    format(c13['c13_out'].iloc[il],'\t',c23['c23_out'].iloc[il],
                           c33['c33_out'].iloc[il],c34['c34_out'].iloc[il],
                           c35['c35_out'].iloc[il],c36['c36_out'].iloc[il],'\n'))
            # row 4-.
            f.write('{0}{1}{2}{1}{3}{1}{4}{1}{5}{1}{6}{7}'.
                    format(c14['c14_out'].iloc[il],'\t',c24['c24_out'].iloc[il],
                           c34['c34_out'].iloc[il],c44['c44_out'].iloc[il],
                           c45['c45_out'].iloc[il],c46['c46_out'].iloc[il],'\n'))
            # row 5-.
            f.write('{0}{1}{2}{1}{3}{1}{4}{1}{5}{1}{6}{7}'.
                    format(c15['c15_out'].iloc[il],'\t',c25['c25_out'].iloc[il],
                           c35['c35_out'].iloc[il],c45['c45_out'].iloc[il],
                           c55['c55_out'].iloc[il],c56['c56_out'].iloc[il],'\n'))
            # row 6-.
            f.write('{0}{1}{2}{1}{3}{1}{4}{1}{5}{1}{6}{7}'.
                    format(c16['c16_out'].iloc[il],'\t',c26['c26_out'].iloc[il],
                           c36['c36_out'].iloc[il],c46['c46_out'].iloc[il],
                           c56['c56_out'].iloc[il],c66['c66_out'].iloc[il],'\n'))
            ## print(il,strain['strain'].iloc[il],sep='\n')
            f.write('{0}{1}'.format('END','\n'*2))
        
        f.close()
## ======================================================================= END79
