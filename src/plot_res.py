#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## to plot results-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Tue Feb  6 21:11:32 CET 2024-.
## last_modify (Fr): Wed Feb  7 16:49:02 CET 2024-.
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
import os

class PlotPredRes():
    '''
         A class to plot results-.
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
    def __init__(self, df):
        self.df=df
        self.feature_var=[var for var in df[0].columns if 'vgrad' in var or '_in' in var]
        self.target_var=[var for var in df[0].columns if '_out' in var]

    def plot_C_VPSC_pred(self, y_pred, ds_name, folder_figs, iruned) ->int:
        ''' plot $C^{VPSC,pred}_{ijkl}$ vs. $\varepsilon$ '''
        for ig, df_model in enumerate(self.df):
            ## set seome matplotlib params
            plt.rcParams.update({'font.size': 6})
            gs=gridspec.GridSpec(6,4) # 24 subfigures-.
            fig=plt.figure(figsize=(20, 10))
            palette=sns.color_palette('mako_r', 4)

            ## feature_in_var-.
            for idx, col in enumerate(self.target_var, start=0):
                ax=fig.add_subplot(gs[idx])  # , sharex=True, sharey=False)
                # true values-.
                var_to_plot=str(col).replace('_out','')
                x_val= df_model.strain
                y_val_vpsc= df_model[str(var_to_plot)+'_out']
                y_val_pred= y_pred[ig][str(var_to_plot)+'_out']
                # 
                plt.scatter(x=x_val,
                            y=y_val_vpsc,
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
                plt.xlim([0.0,x_val.max()])
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
                ' for {0} and IRUN ={1}'.format(str.split(ds_name, '.')[0], iruned[ig], y=0.1)

            fig.text(0.8, 0.1, label, color='r', fontsize=12,
                     horizontalalignment='right', verticalalignment='top',
                     backgroundcolor='1.0'
                     )
            
            plt.show()
            fig.savefig(os.path.join(folder_figs,'Cijkl_VPSC-Pred_Strain_'+
                                     str(iruned[ig])+'.png'), format='png', dpi=100) # -.                                                                  
            ## str(iruned[ig])+'.png'), format='png', dpi=100) # -.

            ## https://stackoverflow.com/questions/14379753/what-does-mean-in-python-function-definitions
            ## print(gt.calcula_elas_anys_coef.__annotations__['return'])
            ## print(gt.calcula_elas_anys_coef.__annotations__['return']['type'])
            ## print(gt.calcula_elas_anys_coef.__annotations__['return']['units'])
            ## print(gt.calcula_elas_anys_coef.__annotations__['return']['docstring'])
        
        return(0)

if __name__=='__main__':
    ## 1-- load dataset-.
    import load_process_ds
    dataset_path='/Users/Fernando/temp/' # in a real life it's read in Config class-.
    dataset_file='db3-SSb.csv' # in a real life it's read in Config class-.
    ds_obj=load_process_ds.Dataset(dataset_path,dataset_file) # in a real life it's read in Config class-.
    ## print(ds_obj)
    ## columns=ds_obj.set_cols('2D') # in a real life it's read in Config class-.
    ## print(columns)
    df_main, feat_var, tar_var=ds_obj.load_ds('2D') # in a real life it's read in Config class-.
    
    '''
    print('{}{}{}'.format('\n'*2,df_main,'\n'*2))
    print('{0}{1}{2}{0}'.format('\n'*2,feat_var, len(feat_var)))
    print('{0}{1}{2}{0}'.format('\n'*2,tar_var, len(tar_var)))
    '''
    
    ## 2-- load ML's model and scalers-.
    import sys
    import load_models_sca
    ## pred= Predict()
    sys.path.append('/Users/Fernando/scratch/elasAnys/2testModels/')
    ## create object
    ms=load_models_sca.ModelScalersStand()
    r_m_s='/Users/Fernando/temp/models/'  # in a real life it's read in Config class-.
    mfn_name='dlModelWithoutHyperOpt.pt'  # in a real life it's read in Config class-.
    mlmodel=ms.load_model(r_m_s, mfn_name)  # in a real life it's read in Config class-.
    ## print(type(mlmodel)) ## MLP.MPL
    # in a real life it's read in Config class-.
    s_f, s_t, s= ms.load_scalers(r_m_s, 'scaler_feat.pkl', 'scaler_targ.pkl','scaler.pkl')

    '''
    print('{0}{1}{0}'.format('\n'*2, mlmodel,))
    print('{0}'.format(s))
    print('{0}'.format(s_f))
    print('{0}'.format(s_t))
    '''

    ## 3- save figs-.
    import save_figs
    ''' main function '''
    sys.path.append('/Users/Fernando/scratch/elasAnys/2testModels/')
    from utils.gen_tools import get_args  as ga
    from utils import gen_tools as gt
    ##
    dataset_path='/Users/Fernando/temp/' # in a real life it's read in Config class-.
    sve_figs=True
    ds_file='db3-SSb.csv' # /Users/Fernando/temp/
    root_save_figs='/Users/Fernando/temp/'
    root='/Users/Fernando/temp/'
    
    ## create object
    sf=save_figs.SaveFigs(sve_figs, ds_file, root_save_figs, root)
    ## print(type(mlmodel)) ## MLP.MPL
    # in a real life it's read in Config class-.
    print('{0}{1}{0}'.format('\n'*2,sf,'\n'*2))
    print('{0}{1}{0}'.format('\n'*2,dir(sf),'\n'*2))
    print('{0}{1}{0}'.format('\n'*2,sf.save_figs,'\n'*2))
    print('{0}{1}{0}'.format('\n'*2,sf.folder_figs,'\n'*2))

    ## 4-- predict-.
    import predict
    pred= predict.Predict(df_main, mlmodel, s_f, s_t)
    print(df_main.columns.to_list)
    ##df_vpsc_main, df_rec_main= pred.pred_recursive_main(1, mlmodel,s_f,s_t)
    df_vpsc_main, df_rec_main, iruned= pred.pred_recursive_main(2)

    ## print(eval(df_vpsc_main[0]))
    ## input(2)
    ## print(df_rec_main)
    ## input(3)

    ## 4-- plot-.
    plotObj= PlotPredRes(df_vpsc_main)
    ret_val=plotObj.plot_C_VPSC_pred(df_rec_main, ds_file, sf.folder_figs, iruned)

