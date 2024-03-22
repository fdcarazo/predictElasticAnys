#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## script to Load and Test DL models trained 
## to predict ELASTIC ANISOTROPY COEFFICIENTS of olivine-.
##q
## DATASET obtained  VPSC simulations. Provided by
## Ph.D. Nestor Cerpa - CNRS - GM - Montpellier University -.
##
## @author: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Sun Apr 23 17:15:35 2023 -.
## last_modify (Arg): Wed Nov 29 10:13:37 CET 2023-.
## last_modify (Fr): Tue Mar 12 18:49:22 CET 2024-.
##
## ======================================================================= END79

## ======================================================================= INI79
## include modulus-.
from pathlib import Path

import utils # IMPORTANT: if I don't write this 2plot/utils/__init__.py is not read-.
import src
## import models
from version import mod_versions as mv

from utils.gen_tools import get_args as ga, write_C_for_mtex as wcfm
from utils import gen_tools as gt

from src.read_config_file import Config as cfg
from src.load_process_ds import Dataset as ds
from src.load_models_sca import ModelScalersStand as ms
from src.save_figs import SaveFigs as sf
from src.predict import Predict as pred
from src.plot_res import PlotPredRes as ppr 
from src.plot_res_statistical import PlotResWithStatistics as prws
from src.bnn_uncertainty import UncertaintyPlots as up
## ======================================================================= END79

## ======================================================================= INI79
def main(config) -> int:
    ''' main: driver '''
    ## import sys
    ## sys.path.append('/Users/Fernando/myGithub/predictElasticAnysNN/models/')
    ## 0- -.
    cfg_obj=cfg(config)

    ## 1- load dataset (2test) as pandas.DataFrame and features and targers var
    ##    names-.
    ds_obj=ds(cfg_obj.ds_path,cfg_obj.ds_file)
    df_main,feat_var,tar_var=ds_obj.load_ds(cfg_obj.dim) 

    '''
    ndf=range(1,6)
    df_main_list=[df_main[df_main['irun']==idx] for idx in ndf]
    import pandas as pd
    df_main=pd.concat(df_main_list,axis=0)
    '''
    ##print(df_main.shape); input(55)
    #################### df_main=df_main.iloc[:200,:]
    
    ## 2- load ML model and scalers (features, targets and all dataset)-.
    ## sys.path.append('/Users/Fernando/scratch/elasAnys/2testModels/')
    ms_obj=ms()
    mlmodel=ms_obj.load_model(cfg_obj.root_mod_sca, cfg_obj.mfn)
    s_f,s_t,s=ms_obj.load_scalers(cfg_obj.root_mod_sca,
                                  cfg_obj.sca_feat,
                                  cfg_obj.sca_targ,
                                  )
    
    ## 3- set path and folder so save_figs, etc., etc.-.
    # cfg_obj.ds_path==root? == nameOfFolderToCreate
    sf_obj= sf(cfg_obj.save,
               cfg_obj.ds_file,
               cfg_obj.dir_save,
               cfg_obj.ds_path
               )

    ## 4- predictions (PandasDF with VPSC/True, NON-RECURSIVE and RECURSIVE predictions)-.
    pred_obj=pred(df_main,mlmodel,s_f,s_t,cfg_obj.irun,cfg_obj.n_nn,feat_var,tar_var)
    df_vpsc_main,df_rec_main,df_pred_main,iruned=pred_obj.pred_recursive_main(cfg_obj.nirun) 

    ## 5- Create Plot_Res Object-.
    ppr_obj=ppr(cfg_obj.dim,df_vpsc_main,df_rec_main,df_pred_main,iruned)
    
    ## BLOCK I- Plots using all dataset-.
    ## 6- PLOT ==> -- RESIDUAL -- errors abs($C^{VPSC/true}-$C^{NON_REC_PRED})$ distribution-.
    ## for test plots I use all VPSC's/IRUN's case. For statistical
    ## plot residual errors (I should create other Object to don't
    ## predict using recursive approach)-.
    df_vpsc_main_1,_,df_pred_main_1,_=pred_obj.pred_recursive_main(1)
    ########## ppr_obj.plot_res(df_vpsc_main_1,df_pred_main_1,cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    ppr_obj.plot_res(df_vpsc_main_1,df_pred_main_1,cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.

    ## 7- save PandasDF with VPSC/True, NON-RECURSIVE and RECURSIVE predictions in files to
    ##    plot using MTEX-.
    ## print(len(df_vpsc_main),len(df_rec_main),len(df_pred_main)); input(55)
    ## print(df_vpsc_main,df_rec_main,df_pred_main), input(55)
    wcfm(df_vpsc_main,df_rec_main,df_pred_main,cfg_obj.dir_save,str.split(cfg_obj.ds_file,'.')[0])

    ## < ================= BLOCK II- Plots using only one VPSC/IRUN  ================= >-.
    ## 8- PLOT ==> Plot -- $C^{VPSC/true,NON_REC_PRED,REC-PRED}$ vs. $\carepsilon$-- -.
    ########## ppr_obj.plot_C_VPSC_pred_vs_def(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    ppr_obj.plot_C_VPSC_pred_vs_def(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.

    ## 9-  PLOT ==> $\varepsilon,eta^{VPSC/true}$ vs. $\bar{\varepsilon}$ and
    ##              $\varepsilon,eta^{VPSC/true}$ vs. $\varepsilon,eta^{NON_REC_PRED,REC_PRED$
    ########## ppr_obj.plot_eps_eta(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    ppr_obj.plot_eps_eta(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    
    ## 10- PLOT ==> CORRELATION -- $C^{VPSC/true} vs. C^{NON_REC_PRED,REC-PRED}$-- -.
    ########## ppr_obj.plot_C_VPSC_vs_C_pred(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    ppr_obj.plot_C_VPSC_vs_C_pred(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    
    ## < ============== BLOCK III- Plots using more than one VPSC/IRUN  ============== >-.
    ## BLOCK III- PLOT_RESULTS_WITH_STATISTICS == prws == (plots with  more than one case,
    ##           mean, std, etc.)-.
    ##  11- Create an object to Plot_Results_With_Statistics-.
    prws_obj=prws(df_main,cfg_obj.nirun,cfg_obj.dim,df_vpsc_main,
                  df_rec_main,df_pred_main,iruned)
    ## 12- PLOT ==> CORRELATION -- $\varepsilon,eta^{VPSC/true}$ vs.
    ##                             $\varepsilon,eta^{NON_REC_PRED,REC_PRED$
    ########## prws_obj.corr_anisitropyc_coefi(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    prws_obj.corr_anisitropyc_coefi(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.

    ## 13- PLOT ==> abs($\Delta C_{ij})=abs($C^{VPSC/true,NON_REC_PRED,REC-PRED}_{ij}$) vs.
    ##              $\varepsilon$-- -.
    ########## prws_obj.plot_diff_C_comp(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    prws_obj.plot_diff_C_comp(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.

    ## 14- PLOT ==> abs(${\varepsilon,\eta})= abs({\varepsilon,\eta}^{VPSC/true})-
    ##              {\varepsilon,\eta}^{NON_REC_PRED,REC-PRED})$ vs. $\varepsilon$-- -.
    ########## prws_obj.plot_diff_eps_eta(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    prws_obj.plot_diff_eps_eta(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.

    ## 15- PLOT ==> $d_{euclidena}($C^{VPSC/true}_{ij},C^{NON_REC_PRED,REC-PRED}_{ij}) vs.
    ##              $\varepsilon$-- -.
    ########## prws_obj.plot_C_dist(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    prws_obj.plot_C_dist(cfg_obj.ds_file,sf_obj.dir_save) # PLOT-PLOT-.
    
    ## 16- For Bayesian predictions-.
    df_vpsc_main,df_rec_main,df_pred_main,iruned=pred_obj.pred_recursive_main(0)
    unc_obj=up(df_vpsc_main,df_rec_main,df_pred_main,iruned) # PLOT-PLOT-.
    ########## unc_obj.plot_with_uncertainty(cfg_obj.ds_file,sf_obj.dir_save,
    ##########                               cfg_obj.quart)  # PLOT-PLOT-.
    unc_obj.plot_with_uncertainty(cfg_obj.ds_file,sf_obj.dir_save,
                                  cfg_obj.quart)  # PLOT-PLOT-.

    
    return 0
## ======================================================================= END79

## ======================================================================= INI79
if __name__ == '__main__':    
    ## 1-4- my auxiliaries methods-.
    ## from utils.gen_tools import *
    ## print(dir())
    config_file=Path(__file__).parent/'config_file.yaml'
    config=ga(config_file)
    
    ## list the name and versions of the main modules used-.
    a=mv(); a.open_save_modules()

    ## call main-.
    val=main(config)
else:
    print('{0} imported as Module'.format(__file__.split('/')[-1]))
