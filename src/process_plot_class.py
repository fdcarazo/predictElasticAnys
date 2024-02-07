#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
##
## SCRIPT: 2 Load and Test ML models to 
## predict ELASTIC ANISOTROPYC COEFFICIENTS of olivine-.
##
## RUNED: in UM's or JeanZay's Cluster-.
##
## DATASET: obtained using VPSC simulations. Provided by
## Ph.D. Nestor Cerpa - CNRS - GM - Montpellier University-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Sun Apr 23 17:15:35 2023 -.
## last_modify (Arg): Wed Nov 29 10:13:37 CET 2023-.
## last_modify (Fr): Mon Feb  5 16:48:30 CET 2024-.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
from numpy import mean, std
from numpy import linalg as LA # 2 calculate euclidean norm-.
import time
## 2save and load scikit-learn/pytorch/etc. models -.
import pickle
## 2find all the pathnames matching a specified pattern according to the
## rules used by the Unix shell-.
import os, psutil, glob
from typing import Dict, List, Union
import yaml
from pathlib import Path
## 2ignore warnings messages-.
import warnings
warnings.filterwarnings('ignore')

## from Scikit-Learn framework -.
## from sklearn.metrics import mean_squared_error, r2_score
## from sklearn.metrics import PredictionErrorDisplay

## 1-2- TORCH-Framework DEEP LEARNINGS MODULES or From torch framework -.
## for PyTorch
import torch as torch

## 1-3- 2utils package-.
from utils.gen_tools import get_args  as ga
from utils import gen_tools as gt

## from models import MLP as MLP
## from models import MLP_transf as MLP_transf
## print(MLP); print(MLP_transf)

class ConfigDsModelFigs():
    def __init__(self, cfg: Dict):
        self.config= cfg # in
        ## out-.
        self.ensemble_dl=self.config['gen_options']['ensemble_dl'] # 2know if the DL model was trained using ensembled approach-.
        self.save_figs=self.config['gen_options']['save_figs'] # 2save or not the figures-.
        self.root_save_figs= self.config['gen_options']['root_save_figs'] # 2set path save figures-.
        self.dim=self.config['gen_options']['dim']
        
        ## 2-2- Paths-.
        self.currentdir=os.path.dirname(os.path.realpath(__file__))
        self.root=os.path.dirname(currentdir)    
        self.root_ds=self.config['datasets']['ds_path']
        
        ## 2-2- Dataset names-.
        ## datasets (used to train and test)-.
        self.ds_path=self.config['datasets']['ds_path']
        self.ds_file1=self.config['datasets']['ds_file1']
        
        ## folder 2 load the ML/DL model-.ou
        ## machine learning file name-.
        self.root_mod_sca=self.config['mlmodel']['ml_path'] ## ml_path= config['mlmodel']['ml_path']
        self.mfn=self.config['mlmodel']['m_l_f_n']
        
        ## scaler/standarizer names-.
        self.sca_feat=self.config['scaler']['scaler_feat'] # used to scale/standarize a FEATURES of the DF-.
        self.sca_targ=self.config['scaler']['scaler_targ'] # used to scale/standarize a TARGETS of the DF-.

        
        def read_config_file(self.config):
        ## ======================================================================= INI79
        ## 2- LOAD DATASETS AND ML and/or DL models-.
        ## 2-1- General options to control the flow of execution-.
        ensemble_dl= config['gen_options']['ensemble_dl'] # 2know if the DL model was trained using ensembled approach-.
        save_figs= config['gen_options']['save_figs'] # 2save or not the figures-.
        root_save_figs= config['gen_options']['root_save_figs'] # 2set path save figures-.
        dim=config['gen_options']['dim']
        
        ## 2-2- Paths-.
        currentdir=os.path.dirname(os.path.realpath(__file__))
        root=os.path.dirname(currentdir)    
        root_ds= config['datasets']['ds_path']
        
        ## 2-2- Dataset names-.
        ## datasets (used to train and test)-.
        ds_path= config['datasets']['ds_path']
        ds_file1= config['datasets']['ds_file1']
        
        ## folder 2 load the ML/DL model-.ou
        ## machine learning file name-.
        root_mod_sca= config['mlmodel']['ml_path'] ## ml_path= config['mlmodel']['ml_path']
        mfn= config['mlmodel']['m_l_f_n']
        
        ## scaler/standarizer names-.
        sca_feat= config['scaler']['scaler_feat'] # used to scale/standarize a FEATURES of the DF-.
        sca_targ= config['scaler']['scaler_targ'] # used to scale/standarize a TARGETS of the DF-.
        
        return ensemble_dl, save_figs, root_save_figs, dim, currentdir, root, root_ds,\
            ds_path, ds_file1, root_mod_sca, mfn, sca_feat, sca_targ

    def read_process_ds(root_ds, ds_file1, dim):
        ## 2-3- Import/load DataSet used to train and test-.
        ## <class 'list'> -DF used to TEST the model-.
        df_name1= glob.glob(os.path.join(str(root_ds), ds_file1))
        if dim='2D':
            cols=['irun','vgrad11', 'vgrad22', 'vgrad12', 'vgrad21','c11_in',
                  'c22_in',, 'c33_in', 'c44_in', 'c55_in', 'c66_in', 'c56_in',
                  'c46_in', 'c36_in', 'c26_in', 'c16_in', 'c45_in', 'c35_in', 
                  'c25_in', 'c15_in', 'c34_in', 'c24_in', 'c14_in', 'c23_in',
                  'c13_in', 'c12_in', 'c11_out', 'c22_out', 'c33_out', 'c44_out',
                  'c55_out', 'c66_out', 'c56_out', 'c46_out', 'c36_out', 'c26_out', 
                  'c16_out', 'c45_out', 'c35_out', 'c25_out', 'c15_out', 'c34_out', 
                  'c24_out', 'c14_out', 'c23_out', 'c13_out', 'c12_out','strain']
        elif dim='3D':
            cols=['irun','vgrad11', 'vgrad22', 'vgrad33', 'vgrad23', 'vgrad13',
                  'vgrad12', 'vgrad32', 'vgrad31', 'vgrad21', 'c11_in', 'c22_in',
                  'c33_in', 'c44_in', 'c55_in', 'c66_in', 'c56_in', 'c46_in',
                  'c36_in', 'c26_in', 'c16_in', 'c45_in', 'c35_in', 'c25_in',
                  'c15_in', 'c34_in', 'c24_in', 'c14_in', 'c23_in', 'c13_in',
                  'c12_in', 'c11_out', 'c22_out', 'c33_out', 'c44_out', 'c55_out',
                  'c66_out', 'c56_out', 'c46_out', 'c36_out', 'c26_out', 'c16_out',
                  'c45_out', 'c35_out', 'c25_out', 'c15_out', 'c34_out', 'c24_out',
                  'c14_out', 'c23_out', 'c13_out', 'c12_out','strain']
        df1= pd.read_csv(df_name1[0], index_col=0, low_memory=False, usecols=cols)

        ## ======================================================================= INI79
        ## 4- TO PROCESS DFs or DataSets loaded-.
        ## WORKING WITH A complete DataFrame -the same used to train DL model-.
        ## take the input (feature) and output (target) variables as strings (in order 
        ## that doesn't depend of the case that are modeling), i.e.
        ## input or feature variables ==> $L_{ij}$ + $C_{ijkl}^{in}$-.
        ## output or target variables ==> $C_{ijkl}^{out}$-.
        target_var= [col for col in df1.columns if '_out' in col]
        ## feature_in_var= [col for col in df1.columns if '_in' in col or 'vgrad' in col]
        feature_in_var= [col for col in df1.columns if '_in' in col or 'vgrad' in col]
        cols_to_consider=[]; irun= ['irun']; strain= ['strain'];
        cols_to_consider=irun+feature_in_var+target_var+strain
        
        df1=df1.loc[:,cols_to_consider]
        df1, n_cols_rem1= gt.remove_cols_zeros(df1) ## it isn't necessary-.
        
        if n_cols_rem1!=0:
            target_var, feature_in_var= [col for col in df1.columns if '_out' in col],\
                [col for col in df1.columns if '_in' in col or 'vgrad' in col]
            # - =======================================================================END79
        return feature_in_var, target_var, df1

    def load_model_sca_std(root_mod_sca, mfn, sca_feat, sca_targ):
        ## 2-4- DL model-.
        ## 2-4-1- check if I've GPU-.
        model_file= root_mod_sca+ mfn
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'
            ## checkpoint = torch.load(load_path, map_location=map_location)
            ## with open(model_file, 'rb') as fm: model= pickle.load(fm)

        ## 2-4-2- Load DL's model-.
        ## with open(model_file, 'rb') as fm: model= torch.load(fm, map_location=torch.device('cpu')) # @ NotEnsemble-.
        ## print(model_file)
        with open(model_file, 'rb') as fm: model= torch.load(fm, map_location=map_location) # @ NotEnsemble-.    
        ## print(model)
        
        ## 2-4-3- load scaler used 2train the DL's modeling-.
        ## 2-4-3-1- scaler/standarizer absolute PATH-.
        scaler_feat_file, scaler_targ_file= root_mod_sca+sca_feat, root_mod_sca+sca_targ

        ## 2-4-3-2 load scaler/standarizer-.
        with open(scaler_feat_file, 'rb') as fs: scaler_feat= pickle.load(fs)
        with open(scaler_targ_file, 'rb') as fs: scaler_targ= pickle.load(fs)
        ## print(scaler_feat), print(scaler_targ)
        ## ======================================================================= END79

        return model, scaler_feat, scaler_targ 

    
    def save_figs(save_figs, ds_file1, root_save_figs, root) -> str:
        ## ======================================================================= INI79
        ## 3- 2save figures-.
        is_folder_exist=True
        if save_figs:
            folder_figs= gt.folder_to_save_figs(ds_file1, root_save_figs, root)
            ## check if folder exist, if not cerate this-.
            is_folder_exist= os.path.exists(folder_figs)
        if not is_folder_exist:
            os.makedirs(folder_figs)
            print('The folder ===< {0}{1}{2} >=== was created.'.format('\t', folder_figs, '\t'))
        else:
            print('The folder ===< {0}{1}{2} >=== exists.'.format('\t', folder_figs, '\t'))
        return folder_figs
    ## ======================================================================= END79

    def plot_C_vpsc_pred(df, in_features, out_targets, scaler_feat, scaler_targ, model) -> int:
        ## ======================a================================================= INI79
        ## 4- 2Plot $C^{out}_{ij}$ tensor componentes-.
        ## 
        ## __FIGURE__7:
        ## 2 plot $C^{out}_{ij}$ VPSC vs. PREDICTED (using recursive approach),
        ## i.e.  $C^{out_{VPSC,PRED_{ITE}}}_{ijkl}$ in function of 
        ## $\bar{\varepsilon}$ for cases extracted from database provided bt NesCer-
        
        ## NOTE: @ gridspec see Arranging multiple Axes in a Figure Matplotlib doc.-.
        n_cases= 1
        df_models, iruned= list(), list()
        for i in range(n_cases):
            ## print(i)
            irun= np.random.randint(df['irun'].min(), df['irun'].max())
            iruned.append(irun)
            ## https://stackoverflow.com/questions/5036700/how-can-you-dynamically-create-variables
            locals()['df_{0}'.format(irun)]= df[df['irun']==irun]
            df_models.append(str('df_{0}'.format(irun)))

        case= '_out'
        df_in= df.loc[:,in_features]
        feat_plus_targ= in_features+ out_targets # features + targets -.
    
        for ig, df_model in enumerate(df_models):
            X_test, y_test= eval(df_model).loc[:,in_features], \
                eval(df_model).loc[:,out_targets] # this database contains STRAIN column-.
            df_temp= eval(df_model).loc[:,feat_plus_targ].copy()
        
            C_in=[var for var in df_temp.columns if '_in' in var]
            L_in=[var for var in df_temp.columns if 'vgrad' in var]
            feat_vars= L_in+ C_in # in_features
        
            ## predict using recursive approach-.
            start_time= time.time()
            ## create a DataFrame as in recursive Approach-.
            df_recur_1= gt.pred_iter_pred_mod(df_temp, feat_vars, target_var, scaler_feat, scaler_targ, model)
            end_time= time.time()
            print('Recursive prediction time {0}'.format(end_time-start_time))

            # remove in_features features/variables from predicted dataset-.
            df_recur_1.drop(columns=df_recur_1.columns.difference(target_var), inplace=True) 
            ## convert PandasDataFrame of predicted values 2 numpy array-.
            ## it isn't necessary beacuse df_recur_1 is a <class 'pandas.core.frame.DataFrame'>-.
            y_pred_df= pd.DataFrame(df_recur_1, columns=target_var) 

            ## set seome matplotlib params
            plt.rcParams.update({'font.size': 6})
            gs= gridspec.GridSpec(6, 4) # 24 subfigures-.
            fig= plt.figure(figsize=(20, 10))
            palette= sns.color_palette('mako_r', 4)
        
            ## feature_in_var-.
            for idx, col in enumerate(target_var, start=0):
                ax= fig.add_subplot(gs[idx])  # , sharex=True, sharey=False)
                # true values-.
                var_to_plot= str(col).replace('_out','')
                x_val= eval(df_model).strain
                y_val_teo= eval(df_model)[str(var_to_plot)+'_out']
                y_val_pred= y_pred_df[str(var_to_plot)+'_out']
                # 
                plt.scatter(x=x_val,
                            y=y_val_teo,
                            s=10,
                            facecolors='none',
                            edgecolor='k',
                            # alpha=0.1,
                            marker='^',
                            # c='blue',
                            label='VPSC'
                            )

                plt.scatter(x=x_val,
                            y=y_val_pred,
                            s=10,
                            facecolors='none',
                            edgecolors='r',
                            # alpha=0.1,
                            marker='o',
                            # c='blue',
                            label='pred'
                            )

                ## plt.legend(loc=3)
                plt.grid()
                plt.xlabel(r'$\bar{\varepsilon}$')
                plt.ylabel('{0}'.format(var_to_plot))
                plt.tight_layout()

                ticks, labels = plt.xticks()
                plt.xticks(ticks[::1], labels[::1])
                plt.xlim([0.0,2.0])
                ## plt.ylim([0.0,280.0])
                ## plt.suptitle(f'Elastic anisotropy tensor componentes '+
                ##              f'predicted using RECURSIVE approach '
                ##              f'for IRUN = {str(df_model)[-1]} in function of '+
                ##              r'$\bar{\varepsilon}$.', y=0.1
                ##              )

            fig.legend(*ax.get_legend_handles_labels(),
                       loc='lower center', ncol=4)
            ## plt.legend(lines, labels, loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
            ##            bbox_transform = plt.gcf().transFigure)
        
            '''
            label= r'($\varepsilon^{VPSC}$, $\varepsilon^{Predicted}_{RECURSIVE}$)'+\
            r' and ($\eta^{VPSC}$ , $\eta^{Predicted}_{RECURSIVE}$) '+\
            r', in function of $\bar{\varepsilon}$ '+ \
            ' for {0} and IRUN ={1}'.format(str.split(ds_file1, '.')[0], iruned[ig], y=0.1)
            '''
        
            label= r'$C_{{ij}}^{{VPSC}}=f(\bar{\varepsilon})$, and '+\
                r'$C_{{ij}}^{{Predicted_{{RECURSIVE}}}}=f(\bar{{\varepsilon}})$'\
                ' for {0} and IRUN ={1}'.format(str.split(ds_file1, '.')[0], iruned[ig], y=0.1)

            fig.text(0.8, 0.1, label, color='r', fontsize=12,
                     horizontalalignment='right', verticalalignment='top',
                     backgroundcolor='1.0'
                     )
        
            plt.show()
            fig.savefig(os.path.join(folder_figs,'Cijkl_VPSC-Pred_'+
                                     str(iruned[ig])+'.png'), format='png', dpi=100) # -.

            ## https://stackoverflow.com/questions/14379753/what-does-mean-in-python-function-definitions
            ## print(gt.calcula_elas_anys_coef.__annotations__['return'])
            ## print(gt.calcula_elas_anys_coef.__annotations__['return']['type'])
            ## print(gt.calcula_elas_anys_coef.__annotations__['return']['units'])
            ## print(gt.calcula_elas_anys_coef.__annotations__['return']['docstring'])
        
        return(0)

