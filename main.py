#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
##
## script to Load and Test ML models runed in a UM's or JeanZay Clusters to 
## predict ELASTIC ANISOTROPYC COEFFICIENTS of olivine-.
##
## DATASET obtained  VPSC simulations. Provided by
## Ph.D. Nestor Cerpa - CNRS - GM - Montpellier University -.
##
## @author: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Sun Apr 23 17:15:35 2023 -.
## last_modify (Arg): Wed Nov 29 10:13:37 CET 2023-.
## last_modify (Fr): Wed Feb  7 12:18:38 CET 2024-.
##
## ======================================================================= END79

## ======================================================================= INI79
## include modulus-.
from pathlib import Path

import utils # IMPORTANT: if I don't write this 2plot/utils/__init__.py is not read-.
import src
## import models
from version import mod_versions as mv

from utils.gen_tools import get_args  as ga
from utils import gen_tools as gt

from src.read_config_file import Config as cfg
from src.load_process_ds import Dataset as ds
from src.load_models_sca import ModelScalersStand as ms
from src.save_figs import SaveFigs as sf
from src.predict import Predict as pred
from src.plot_res import PlotPredRes as ppr
## ======================================================================= END79

## ======================================================================= INI79
def main(config) -> int:
    ''' main: driver '''
    ## 0- -.
    cfg_obj=cfg(config)

    ## 1- -.
    ds_obj=ds(cfg_obj.ds_path, cfg_obj.ds_file)
    df_main, feat_var, tar_var=ds_obj.load_ds(cfg_obj.dim) 

    ## 2- -.
    ## sys.path.append('/Users/Fernando/scratch/elasAnys/2testModels/')
    ms_obj=ms()
    mlmodel=ms_obj.load_model(cfg_obj.root_mod_sca, cfg_obj.mfn)
    s_f, s_t, s=ms_obj.load_scalers(cfg_obj.root_mod_sca,
                                    cfg_obj.sca_feat,
                                    cfg_obj.sca_targ,
                                    )
    
    ## 3- -.
    # cfg_obj.ds_path==root? == nameOfFolderToCreate
    sf_obj= sf(cfg_obj.save_figs,
               cfg_obj.ds_file,
               cfg_obj.dir_save_figs,
               cfg_obj.ds_path
               )

    ## 4- -.
    pred_obj=pred(df_main,mlmodel,s_f, s_t,cfg_obj.irun)
    df_vpsc_main, df_rec_main, iruned= pred_obj.pred_recursive_main(1)

    ## 5- -.
    ppr_obj=ppr(df_vpsc_main)
    ret_val=ppr_obj.plot_C_VPSC_pred(df_rec_main,
                                     cfg_obj.ds_file,
                                     sf_obj.folder_figs,
                                     iruned
                                     )

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
