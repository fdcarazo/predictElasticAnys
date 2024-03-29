#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## rergession with uncertantiy-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Fri Mar 15 08:18:57 CET 2024-.
## last_modify (Fr): -.
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
import time
import os
import tqdm

class UncertaintyPlots():
    '''
         A class to plot regression with uncertantiy quantification-.
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
    def __init__(self,df,df_pred_rec,df_pred,iruned,dim,idx_o):
        ''' constructor '''
        self.df_true=df[0]; self.df_pred=df_pred; self.df_pred_rec=df_pred_rec
        self.iruned=iruned
        self.target_var=[col for col in self.df_true.columns if '_out' in col]
        self.dim=dim
        self.idx_o=idx_o
        
    def plot_with_uncertainty(self,ds_name:str,dir_save:str,quart:int):
        '''
        plot $C^{VPSC-true}_{ij} $$C^{NON-REWC_pred}_{ij}$ and $C^{REC_pred}_{ij}$
        '''
        y_mean_pred=np.mean(self.df_pred,axis=0)
        y_mean_pred_rec=np.mean(self.df_pred_rec,axis=0)
        
        y_std_pred=np.std(self.df_pred,axis=0)
        y_std_pred_rec=np.std(self.df_pred_rec,axis=0)

        nr,nc=6,4 # if str.lower(dim)=='2d' else (7,6) if dim=='3d' else (None,None)
        plt.rcParams.update({'font.size': 6}); gs=gridspec.GridSpec(nr,nc)
        fig=plt.figure(figsize=(20, 10))
        sns.color_palette('mako_r',4)
        
        ## feature_in_var-.
        for idx, col in enumerate(self.target_var,start=0):
            ax=fig.add_subplot(gs[idx])  # , sharex=True, sharey=False)
            # true values-.
            var_to_plot=str(col).replace('_out','')
            x=self.df_true.strain
            
            ## VPSC/True-.
            plt.scatter(x=x,y=self.df_true.loc[:,str(col)],s=10,facecolors='none',
                        edgecolor='k',marker='^',label='VPSC/true'
                        # alpha=0.1, c='blue',
                        )
            ## NON-RECURSIVE pred-.
            plt.scatter(x=x,y=y_mean_pred[:,idx],s=10,facecolors='none',
                        edgecolors='r',marker='o',label='NON-RECURSIVE prediction'
                        # alpha=0.1,c='blue',
                        )
            plt.fill_between(x,
                             y_mean_pred[:,idx]+quart*y_std_pred[:,idx],
                             y_mean_pred[:,idx]-quart*y_std_pred[:,idx], 
                             alpha=0.5,label='Epistemic uncertainty NON-RECURSIVE'
                             )
            ## RECURSIVE pred-.
            plt.scatter(x=x,y=y_mean_pred_rec[:,idx],s=10,facecolor='None',
                        edgecolor='b',marker='*',label='RECURSIVE prediction'
                        # alpha=0.1, c='blue',
                        )
            plt.fill_between(x,
                             y_mean_pred_rec[:,idx]+quart*y_std_pred_rec[:,idx],
                             y_mean_pred_rec[:,idx]-quart*y_std_pred_rec[:,idx], 
                             alpha=0.5,label='Epistemic uncertainty RECURSIVE'
                             )                        
            ## plt.legend(loc=3)
            plt.grid()
            plt.xlabel(r'$\bar{\varepsilon}$')
        
            col=str(col).replace('_out','')
            plt.ylabel(r'$\Delta$'+
                       '{0}'.format(col)
                       )
            plt.tight_layout()
        
            ticks, labels=plt.xticks()
            plt.xticks(ticks[::1], labels[::1])
            ## plt.xlim([0.0,2.0])
        
            fig.legend(*ax.get_legend_handles_labels(),
                       loc='lower center', ncol=4)
        
        ## problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        
        label= r'$C_{{ij}}^{{VPSC}}-C_{{ij}}^{{REC-PRED}}-C_{{ij}}^{{NON-REC-PRED}}= '\
            r'f(\bar{\varepsilon})$'## +\
            ##    ' , for {0}'.format(self.i_run)
            ## r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
            ## r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
                
        fig.text(0.7,0.12,label,color='r',fontsize=16,
                 horizontalalignment='right',verticalalignment='top',
                 backgroundcolor='1.0')
        
        plt.show()
        fig.savefig(os.path.join(dir_save,'Cij_withUncertainty.png'),format='png',dpi=100)
        
    def plot_with_uncertainty_1(self,ds_name:str,dir_save:str,quart:int):
        '''
        plot $C^{VPSC-true}_{ij} $$C^{NON-REWC_pred}_{ij}$ and $C^{REC_pred}_{ij}$
        '''
        y_mean_pred=np.mean(self.df_pred,axis=0)
        y_mean_pred_rec=np.mean(self.df_pred_rec,axis=0)
        
        y_std_pred=np.std(self.df_pred,axis=0)
        y_std_pred_rec=np.std(self.df_pred_rec,axis=0)

        ## feature_in_var-.
        ## set seome matplotlib params
        dict_plt_rcParams,gs,fig,plette=self.set_plot_options_all_C(self.dim)
        plt.rcParams.update(dict_plt_rcParams)
        nr,nc=6,6
        fig,axes=plt.subplots(nrows=nr,ncols=nc,figsize=(20,10))
        idx=0
        for i in range(nr):
            for j in range(nc):
                ## distributions of errros/residuals-.
                if i<=j:
                    ax=fig.add_subplot(axes[i][j])  # , sharex=True, sharey=False)
                    col=self.target_var[self.idx_o[idx]]
                    var_to_plot=str(col).replace('_out','')
                    x=self.df_true.strain

                    # true values-.
                    ## VPSC/True-.
                    plt.scatter(x=x,y=self.df_true.loc[:,str(col)],s=10,facecolors='none',
                                edgecolor='k',marker='^',label='VPSC/true'
                                # alpha=0.1, c='blue',
                                )
                    ## NON-RECURSIVE pred-.
                    plt.scatter(x=x,y=y_mean_pred[:,self.idx_o[idx]],s=10,facecolors='none',
                                edgecolors='r',marker='o',label='NON-RECURSIVE prediction'
                                # alpha=0.1,c='blue',
                                )
                    plt.fill_between(x,
                                     y_mean_pred[:,self.idx_o[idx]]+quart*y_std_pred[:,idx],
                                     y_mean_pred[:,self.idx_o[idx]]-quart*y_std_pred[:,idx], 
                                     alpha=0.5,label='Epistemic uncertainty NON-RECURSIVE'
                                     )
                    ## RECURSIVE pred-.
                    plt.scatter(x=x,y=y_mean_pred_rec[:,self.idx_o[idx]],s=10,facecolor='None',
                                edgecolor='b',marker='*',label='RECURSIVE prediction'
                                # alpha=0.1, c='blue',
                                )
                    plt.fill_between(x,
                                     y_mean_pred_rec[:,self.idx_o[idx]]+quart*y_std_pred_rec[:,idx],
                                     y_mean_pred_rec[:,self.idx_o[idx]]-quart*y_std_pred_rec[:,idx], 
                                     alpha=0.5,label='Epistemic uncertainty RECURSIVE'
                                     )                        
                    ## plt.legend(loc=3)
                    plt.grid()
                    plt.xlabel(r'$\bar{\varepsilon}$')
                    
                    col=str(col).replace('_out','')
                    ## plt.ylabel(r'$\Delta$'+'{0}'.format(col))
                    plt.ylabel('{0}'.format(col))
                    plt.tight_layout()
                    
                    ticks, labels=plt.xticks()
                    plt.xticks(ticks[::1], labels[::1])
                    ## plt.xlim([0.0,2.0])
                    
                    fig.legend(*ax.get_legend_handles_labels(),
                               loc='lower center', ncol=4)
                    
                    idx+=1
                else:
                    axes[i][j].remove()

                    
        ## problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        
        label= r'$C_{{ij}}^{{VPSC}}-C_{{ij}}^{{REC-PRED}}-C_{{ij}}^{{NON-REC-PRED}}= '\
            r'f(\bar{\varepsilon})$'## +\
            ##    ' , for {0}'.format(self.i_run)
            ## r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
            ## r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
                
        fig.text(0.7,0.12,label,color='r',fontsize=16,
                 horizontalalignment='right',verticalalignment='top',
                 backgroundcolor='1.0')
        
        plt.show()
        fig.savefig(os.path.join(dir_save,'Cij_withUncertainty.png'),format='png',dpi=100)
        
    def set_plot_options_all_C(self,dim):
        ''' set genreal option to plot all C tensor component values-.'''
        ##plt.set_context('paper',rc={'font.size':4,
        ##                            'axes.titlesize':4,
        ##                            'axes.labelsize':4})            
        nr,nc=(6,6) if str.lower(dim)=='2d' else (7,6) if dim=='3d' else (None,None)
        return {'font.size': 6}, gridspec.GridSpec(nr,nc),\
            plt.figure(figsize=(20, 10)),sns.color_palette('mako_r',4)
