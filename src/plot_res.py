#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## to plot RESULTS: all $C^{VPSC/true-predicted}_{ij}$-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Tue Feb  6 21:11:32 CET 2024-.
## last_modify (Fr): Thu Mar 14 09:21:08 CET 2024-.
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
import os

from sklearn.metrics import mean_squared_error,r2_score

from utils.elastic_calcs import calcula_elas_anys_coef as ceac
from utils.gen_tools import adjust_line as al, eps_eta_rmse as eermse

class PlotPredRes():
    '''
         A class to plot RESULTS: all $C^{VPSC/true-predicted}_{ij}$-.
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
    def __init__(self,dim:str,df_true:list,df_pred_rec:list,df_pred:list,iruned:list,
                 feat_var:list,target_vars:list,idx_o:list):
        self.dim=dim # problem dimension ('2d' or '3d')-,
        self.df_true=df_true # list with Pandas.DataFrames, len(df_true)=cfg.nirun-.
        self.df_pred_rec=df_pred_rec # list with Pandas.DataFrames RECURSIVE, len(df_pred)=cfg.nirun-.
        self.df_pred=df_pred # list with Pandas.DataFrames NON-RECURSIVE, len(df_pred)=cfg.nirun-.
        self.iruned=iruned # list of Pandas.DataFrames, len(iruned)=cfg.nirun-.
        self.feature_var=feat_var
        self.target_var=target_vars
        self.idx_o=idx_o # list to write $C^{out}_{ij}$ in order way in the upper right triangle-.
        ## self.feature_var=[var for var in df_true[0].columns if 'vgrad' in var or '_in' in var]
        ## self.target_var=[var for var in df_true[0].columns if '_out' in var]
        ## I choose one of the dfs (VPSC/IRUNs) to do the plots defined in this class-.
        self.i_run=np.random.randint(0,len(self.df_true))
        self.df=[self.df_true[self.i_run]]
        ## print(self.i_run,df,sep='\n'), input(0)
        
    def plot_C_VPSC_pred_vs_def(self,ds_name:str,dir_save:str):
        ''' plot $C^{VPSC,pred}_{ijkl}$ vs. $\varepsilon$ '''
        ## for ig, df_model in enumerate(self.df_true): # @1 decomment this line to plot more than one IRUN-.
        for ig, df_model in enumerate(self.df):
            ## set seome matplotlib params
            dict_plt_rcParams,gs,fig,plette=self.set_plot_options_all_C(self.dim)
            plt.rcParams.update(dict_plt_rcParams)
            ## print(self.target_var); input('44')
            ## feature_in_var-.
            for idx, col in enumerate(self.target_var,start=0):
                ax=fig.add_subplot(gs[idx])  # , sharex=True, sharey=False)
                # true values-.
                var_to_plot=str(col).replace('_out','')
                x=df_model.strain
                y_true=df_model[str(var_to_plot)+'_out']
                ## y_pred_rec=self.df_pred_rec[ig][str(var_to_plot)+'_out'] # @1-.
                y_pred_rec=self.df_pred_rec[self.i_run][str(var_to_plot)+'_out'] # RECURSIVE-.
                ## y_pred=self.df_pred[ig][str(var_to_plot)+'_out'] # @1-.
                y_pred=self.df_pred[self.i_run][str(var_to_plot)+'_out'] # NON_RECURSIVE-.
                
                plt.scatter(x=x,y=y_true,s=10,facecolors='none',
                            edgecolor='k',marker='^',label='VPSC/true'
                            # alpha=0.1, c='blue',
                            )
                plt.scatter(x=x,y=y_pred_rec,s=10,facecolors='none',
                            edgecolors='r',marker='o',label='recursive_prediction'
                            # alpha=0.1,c='blue',
                            )
                plt.scatter(x=x,y=y_pred,s=10,facecolor='None',
                            edgecolor='b',marker='*',label='non_recursive_prediction'
                            # alpha=0.1, c='blue',
                            )
                
                ## plt.legend(loc=3)
                plt.grid()
                plt.xlabel(r'$\bar{\varepsilon}$')
                plt.ylabel('{0}'.format(var_to_plot))
                plt.tight_layout()
                
                ticks,labels=plt.xticks()
                plt.xticks(ticks[::1], labels[::1])
                plt.xlim([0.0,x.max()])
                ## plt.ylim([0.0,280.0])
                ## plt.suptitle(f'Elastic anisotropy tensor componentes '+
                ##              f'predicted using RECURSIVE approach '
                ##              f'for IRUN = {str(df_model)[-1]} in function of '+
                ##              r'$\bar{\varepsilon}$.', y=0.1
                ##              )
                    
                fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=4)
                ## plt.legend(lines, labels, loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
                ##            bbox_transform = plt.gcf().transFigure)
                
            '''
            label= r'($\varepsilon^{VPSC}$, $\varepsilon^{Predicted}_{RECURSIVE}$)'+\
            r' and ($\eta^{VPSC}$ , $\eta^{Predicted}_{RECURSIVE}$) '+\
            r', in function of $\bar{\varepsilon}$ '+ \
            ' for {0} and IRUN ={1}'.format(str.split(ds_file1, '.')[0], iruned[ig], y=0.1)
            '''
            
            label= r'$C_{{ij}}^{{VPSC/true}}=f(\bar{\varepsilon})$, and '+\
                r'$C_{{ij}}^{{predicted_{{RECURSIVE}}}}=f(\bar{{\varepsilon}})$'\
                ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.i_run,y=0.1)
            ## ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.iruned[ig],y=0.1)  # @1-.
            
            fig.text(0.8,0.1,label,color='r',fontsize=12,
                     horizontalalignment='right',verticalalignment='top',
                     backgroundcolor='1.0'
                     )
            
            plt.show()
            fig.savefig(os.path.join(dir_save,'Cijkl_VPSC-Pred_Strain_'+
                                     str(self.i_run)+'.png'),
                        ##str(self.iruned[ig])+'.png'),  # @1-.
                        format='png', dpi=100) # -.
            
        ## str(iruned[ig])+'.png'), format='png', dpi=100) # -.
        
        ## https://stackoverflow.com/questions/14379753/what-does-mean-in-python-function-definitions
        ## print(gt.calcula_elas_anys_coef.__annotations__['return'])
        ## print(gt.calcula_elas_anys_coef.__annotations__['return']['type'])
        ## print(gt.calcula_elas_anys_coef.__annotations__['return']['units'])
        ## print(gt.calcula_elas_anys_coef.__annotations__['return']['docstring'])
        
    def plot_C_VPSC_pred_vs_def_1(self,ds_name:str,dir_save:str):
        ''' plot $C^{VPSC,pred}_{ijkl}$ vs. $\varepsilon$ '''
        
        for ig, df_model in enumerate(self.df):
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
                        var=str(col).replace('_out','')
                        x=self.df[0].strain
                        y_true=self.df[0][str(var)+'_out']
                        ## y_pred_rec=self.df_pred_rec[ig][str(var)+'_out'] # @1-.
                        y_pred_rec=self.df_pred_rec[self.i_run][str(var)+'_out'] # RECURSIVE-.
                        ## y_pred=self.df_pred[ig][str(var_to_plot)+'_out'] # @1-.
                        y_pred=self.df_pred[self.i_run][str(var)+'_out'] # NON_RECURSIVE-.
                        
                        plt.scatter(x=x,y=y_true,s=10,facecolors='none',
                                    edgecolor='k',marker='^',label='VPSC/true'
                                    # alpha=0.1, c='blue',
                                    )
                        
                        plt.scatter(x=x,y=y_pred_rec,s=10,facecolors='none',
                                    edgecolors='r',marker='o',label='recursive_prediction'
                                    # alpha=0.1,c='blue',
                                    )
                        plt.scatter(x=x,y=y_pred,s=10,facecolor='None',
                                    edgecolor='b',marker='*',label='non_recursive_prediction'
                                    # alpha=0.1, c='blue',
                                    )
                        
                        ## plt.legend(loc=3)
                        plt.grid()
                        ## plt.legend(loc=2)
                        plt.tight_layout()
                        plt.xlabel(r'$\bar{\varepsilon}$')
                        plt.ylabel('{0}'.format(var))
                        
                        ticks,labels=plt.xticks()
                        plt.xticks(ticks[::1], labels[::1])
                        plt.xlim([0.0,x.max()])
                        
                        ## plt.ylim([0.0,280.0])
                        ## plt.suptitle(f'Elastic anisotropy tensor componentes '+
                        ##              f'predicted using RECURSIVE approach '
                        ##              f'for IRUN = {str(df_model)[-1]} in function of '+
                        ##              r'$\bar{\varepsilon}$.', y=0.1
                        ##              )
                        
                        idx+=1
                        
                    else:
                        axes[i][j].remove()
                
            fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=4)
            ## plt.legend(lines, labels, loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
            ##            bbox_transform = plt.gcf().transFigure)
            
            '''
            label= r'($\varepsilon^{VPSC}$, $\varepsilon^{Predicted}_{RECURSIVE}$)'+\
            r' and ($\eta^{VPSC}$ , $\eta^{Predicted}_{RECURSIVE}$) '+\
            r', in function of $\bar{\varepsilon}$ '+ \
            ' for {0} and IRUN ={1}'.format(str.split(ds_file1, '.')[0], iruned[ig], y=0.1)
            '''
            
            label= r'$C_{{ij}}^{{VPSC/true}}=f(\bar{\varepsilon})$, and '+\
                r'$C_{{ij}}^{{predicted_{{RECURSIVE}}}}=f(\bar{{\varepsilon}})$'\
                ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.i_run,y=0.1)
                ## ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.iruned[ig],y=0.1)  # @1-.
            
            fig.text(0.8,0.1,label,color='r',fontsize=12,
                     horizontalalignment='right',verticalalignment='top',
                     backgroundcolor='1.0'
                     )
            
            plt.show()
            fig.savefig(os.path.join(dir_save,'Cijkl_VPSC-Pred_Strain_'+
                                     str(self.i_run)+'.png'),
                        ##str(self.iruned[ig])+'.png'),  # @1-.
                        format='png', dpi=100) # -.
            
            ## str(iruned[ig])+'.png'), format='png', dpi=100) # -.
            ## https://stackoverflow.com/questions/14379753/what-does-mean-in-python-function-definitions
            ## print(gt.calcula_elas_anys_coef.__annotations__['return'])
            ## print(gt.calcula_elas_anys_coef.__annotations__['return']['type'])
            ## print(gt.calcula_elas_anys_coef.__annotations__['return']['units'])
            ## print(gt.calcula_elas_anys_coef.__annotations__['return']['docstring'])
        
    ## distribution of VPSC/true and predicted $C^{out}_{ijkl}$ residual
    ## in all VPSC/IRUNs dataset (only with NON-RECURSIVE predictions)-.
    ## @staticmethod
    def plot_res(self,df,df_pred,ds_name,dir_save):
        ''' plot distribution of $C^{VPSC}_{ijkl}$ - $C^{pred}_{ijkl}$$ '''
        ## for ig, df_model in enumerate(self.df_true): #  # @1 decomment this line to plot more than one IRUN-.
        for ig, df_model in enumerate(self.df):
            ## set some matplotlib params-.
            dict_plt_rcParams,gs,fig,plette=self.set_plot_options_all_C(self.dim)
            plt.rcParams.update(dict_plt_rcParams)
            
            ## distributions of errros/residuals-.
            for idx, col in enumerate(self.target_var,start=0):
                ax=fig.add_subplot(gs[idx])  # , sharex=True, sharey=False)
                var=str(col).replace('_out','')

                # if I don't applied np.round the plot fail-.
                ax=sns.histplot(np.round((df[0][col].to_numpy(dtype=float)- 
                                          df_pred[0][col].to_numpy(dtype=float)),2),
                                kde=True,label=str(col),legend=True,stat='percent')
                plt.legend(loc=2)
                plt.grid()
                plt.tight_layout()
                
                ## plt.suptitle('ERROR DISTRIBUTIONS BETWEEN THEORETICAL vs. PREDICTED elastic  '+\
                ##              'anisotropy tensor components In NON-ITERATIVE APPROACH')
                ## label=var
            ## print(fig.get_size_inches()[1])
            #### plt.text(0.9, 0.25, label, color='g', fontsize=10,
            ####          horizontalalignment='right', verticalalignment='top',
            ####          backgroundcolor='1.0', transform=plt.gca().transAxes
            ####          )

            label=r'$C_{{ij}}^{{VPSC/true}}=f(\bar{\varepsilon})$-'+\
                r'$C_{{ij}}^{{predicted_{{RECURSIVE}}}}$'\
                ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.i_run,y=0.1)
            ## ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.iruned[ig],y=0.1) # @1-.
            
            fig.text(0.8,0.1,label,color='r',fontsize=12,
                     horizontalalignment='right',verticalalignment='top',
                     backgroundcolor='1.0'
                     )

            plt.show()
            ##fig.savefig(os.path.join(dir_save,'residual_'+str(self.iruned[ig])+'.png'), # @1-.
            fig.savefig(os.path.join(dir_save,'residual_'+str(self.i_run)+'.png'),
                        format='png', dpi=100) # -.
    
    ## distribution of VPSC/true and predicted $C^{out}_{ijkl}$ residual
    ## in all VPSC/IRUNs dataset (only with NON-RECURSIVE predictions)-.
    ## @staticmethod
    def plot_res_1(self,df,df_pred,ds_name,dir_save):
        ''' plot distribution of $C^{VPSC}_{ijkl}$ - $C^{pred}_{ijkl}$$ '''
        ## for ig, df_model in enumerate(self.df_true): #  # @1 decomment this line to plot more than one IRUN-.
        for ig, df_model in enumerate(self.df):
            ## set seome matplotlib params
            dict_plt_rcParams,gs,fig,plette=self.set_plot_options_all_C(self.dim)
            plt.rcParams.update(dict_plt_rcParams)
            nr,ncol=6,6
            fig,axes=plt.subplots(nrows=6,ncols=6,figsize=(20,10))
            idx=0
            for i in range(nr):
                for j in range(ncol):
                    ## distributions of errros/residuals-.
                    if i<=j:
                        ax=fig.add_subplot(axes[i][j])  # , sharex=True, sharey=False)
                        col=self.target_var[self.idx_o[idx]]
                        var=str(col).replace('_out','')
                        
                        # if I don't applied np.round the plot fail-.
                        ax=sns.histplot(np.round((df[0][col].to_numpy(dtype=float)- 
                                                  df_pred[0][col].to_numpy(dtype=float)),2),
                                        kde=True,label=str(col),legend=True,stat='percent')
                        plt.legend(loc=2)
                        plt.grid()
                        plt.tight_layout()
                        idx+=1
                    else:
                        axes[i][j].remove()
                
                ## plt.suptitle('ERROR DISTRIBUTIONS BETWEEN THEORETICAL vs. PREDICTED elastic  '+\
                ##              'anisotropy tensor components In NON-ITERATIVE APPROACH')
                ## label=var
            ## print(fig.get_size_inches()[1])
            #### plt.text(0.9, 0.25, label, color='g', fontsize=10,
            ####          horizontalalignment='right', verticalalignment='top',
            ####          backgroundcolor='1.0', transform=plt.gca().transAxes
            ####          )

            label=r'$C_{{ij}}^{{VPSC/true}}=f(\bar{\varepsilon})$-'+\
                r'$C_{{ij}}^{{predicted_{{RECURSIVE}}}}$'\
                ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.i_run,y=0.1)
            ## ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.iruned[ig],y=0.1) # @1-.
            
            fig.text(0.8,0.1,label,color='r',fontsize=12,
                     horizontalalignment='right',verticalalignment='top',
                     backgroundcolor='1.0'
                     )

            plt.show()
            ##fig.savefig(os.path.join(dir_save,'residual_'+str(self.iruned[ig])+'.png'), # @1-.
            fig.savefig(os.path.join(dir_save,'residual_'+str(self.i_run)+'.png'),
                        format='png', dpi=100) # -.
            
    def plot_C_VPSC_vs_C_pred(self,ds_name:str,dir_save:str):    
        '''
        plot $C^{VPSC/true}_{ijkl}$ vs. $C^{pred}_{ijkl}$-.
        to study the correlation between theoretical and
        predicted $C^{out}_{ijkl}-.
        '''
        ## for ig, df_model in enumerate(self.df_true): #  # @1 decomment this line to plot more than one IRUN-.
        for ig, df_model in enumerate(self.df):
            ## set seome matplotlib params
            dict_plt_rcParams,gs,fig,plette=self.set_plot_options_all_C(self.dim)
            plt.rcParams.update(dict_plt_rcParams)
            ## feature_in_var-.
            for idx, col in enumerate(self.target_var,start=0):
                ax=fig.add_subplot(gs[idx])  # , sharex=True, sharey=False)
                # true values-.
                var_to_plot=str(col).replace('_out','')
                
                y_true=df_model[str(var_to_plot)+'_out']
                ## y_pred_rec=self.df_pred_rec[ig][str(var_to_plot)+'_out']  # @1-.
                y_pred_rec=self.df_pred_rec[self.i_run][str(var_to_plot)+'_out']                
                ## y_pred=self.df_pred[ig][str(var_to_plot)+'_out']  # @1-.
                y_pred=self.df_pred[self.i_run][str(var_to_plot)+'_out']
            
                plt.scatter(x=y_true,y=y_pred,s=10,facecolors='none',
                            edgecolor='k',marker='^',label='VPSC/true-pred/NON-REC'
                            # alpha=0.1, c='blue',
                            )
                plt.scatter(x=y_true,y=y_pred_rec,s=10,facecolors='none',
                            edgecolors='r',marker='o',label='VPSC/true-pred/RECURSIVE'
                            # alpha=0.1,c='blue',
                            )

                # adjust straight line (for non recursive aprroach)-.
                pearR=np.corrcoef(y_true,y_pred)[1,0]
                A=np.vstack([y_true,np.ones(len(y_true))]).T
                m,c=np.linalg.lstsq(A,y_pred)[0]
                color='b'
                plt.plot(y_true,y_true*m+c,color=color,label='Fit -- r = %6.4f'%(pearR))
                # adjust straight line (for recursive aprroach)-.
                pearR=np.corrcoef(y_true,y_pred_rec)[1,0]
                A=np.vstack([y_true,np.ones(len(y_true))]).T
                m,c=np.linalg.lstsq(A,y_pred_rec)[0]
                color='g'
                plt.plot(y_true,y_true*m+c,color=color,label='Fit -- r = %6.4f'%(pearR))

                plt.legend(loc=2)
                ## plt.plot(color='blue', label='Fit %6s, r = %6.2e'%(color,pearR))
                ## plt.plot(label='r = %6.2e'%(pearR))
                plt.grid()
                plt.tight_layout()
                
                ##plt.suptitle('VPSC vs. PREDICTED elastic anyiotropy tensor components '+\
                ##             'In NON-ITERATIVE APPROACH for IRUN ={0}'.format(1))
                
                ## label= var_to_plot
                ## print(fig.get_size_inches()[1])
                ##plt.text(0.9, 0.25, label, color='g', fontsize=10,
                ##         horizontalalignment='right', verticalalignment='top',
                ##         backgroundcolor='1.0', transform=plt.gca().transAxes
                ##         )
            plt.show()
            fig.savefig(os.path.join(dir_save,'C_VPSC_true-Pred_REC_NONREC'+
                                     str(self.i_run)+'.png'),format='png',
                                     ## str(self.iruned[ig])+'.png'),format='png',  # @1-.
                        dpi=100) # -.
    
    def plot_C_VPSC_vs_C_pred_1(self,ds_name:str,dir_save:str):    
        '''
        plot $C^{VPSC/true}_{ijkl}$ vs. $C^{pred}_{ijkl}$-.
        to study the correlation between theoretical and
        predicted $C^{out}_{ijkl}-.
        '''
        ## for ig, df_model in enumerate(self.df_true): #  # @1 decomment this line to plot more than one IRUN-.

        for ig, df_model in enumerate(self.df):
            ## set seome matplotlib params
            dict_plt_rcParams,gs,fig,plette=self.set_plot_options_all_C(self.dim)
            plt.rcParams.update(dict_plt_rcParams)
            nr,ncol=6,6
            fig,axes=plt.subplots(nrows=6,ncols=6,figsize=(20,10))
            idx=0
            for i in range(nr):
                for j in range(ncol):
                    ## distributions of errros/residuals-.
                    if i<=j:
                        ax=fig.add_subplot(axes[i][j])  # , sharex=True, sharey=False)
                        col=self.target_var[self.idx_o[idx]]
                        var=str(col).replace('_out','')
                        x=self.df[0].strain
                        y_true=self.df[0][str(var)+'_out']
                        ## y_pred_rec=self.df_pred_rec[ig][str(var)+'_out'] # @1-.
                        y_pred_rec=self.df_pred_rec[self.i_run][str(var)+'_out'] # RECURSIVE-.
                        ## y_pred=self.df_pred[ig][str(var_to_plot)+'_out'] # @1-.
                        y_pred=self.df_pred[self.i_run][str(var)+'_out'] # NON_RECURSIVE-.
                        
                        plt.scatter(x=y_true,y=y_pred,s=10,facecolors='none',
                                    edgecolor='k',marker='^',label='VPSC/true-pred/NON-REC'
                                    # alpha=0.1, c='blue',
                                    )
                        plt.scatter(x=y_true,y=y_pred_rec,s=10,facecolors='none',
                                    edgecolors='r',marker='o',label='VPSC/true-pred/RECURSIVE'
                                    # alpha=0.1,c='blue',
                                    )
                        
                        # adjust straight line (for non recursive aprroach)-.
                        pearR=np.corrcoef(y_true,y_pred)[1,0]
                        A=np.vstack([y_true,np.ones(len(y_true))]).T
                        m,c=np.linalg.lstsq(A,y_pred)[0]
                        color='b'
                        plt.plot(y_true,y_true*m+c,color=color,label='Fit -- r = %6.4f'%(pearR))
                        # adjust straight line (for recursive aprroach)-.
                        pearR=np.corrcoef(y_true,y_pred_rec)[1,0]
                        A=np.vstack([y_true,np.ones(len(y_true))]).T
                        m,c=np.linalg.lstsq(A,y_pred_rec)[0]
                        color='g'
                        plt.plot(y_true,y_true*m+c,color=color,label='Fit -- r = %6.4f'%(pearR))
                        
                        ## plt.legend(loc=2)
                        ## plt.plot(color='blue', label='Fit %6s, r = %6.2e'%(color,pearR))
                        ## plt.plot(label='r = %6.2e'%(pearR))
                        plt.grid()
                        plt.tight_layout()
                        
                        plt.xlabel(r'$\bar{\varepsilon}$')
                        plt.ylabel('{0}'.format(var))
                        
                        ## ticks,labels=plt.xticks()
                        ## plt.xticks(ticks[::1], labels[::1])
                        
                        ##plt.suptitle('VPSC vs. PREDICTED elastic anyiotropy tensor components '+\
                            ##             'In NON-ITERATIVE APPROACH for IRUN ={0}'.format(1))
                        
                        ## label= var_to_plot
                        ## print(fig.get_size_inches()[1])
                        ##plt.text(0.9, 0.25, label, color='g', fontsize=10,
                        ##         horizontalalignment='right', verticalalignment='top',
                        ##         backgroundcolor='1.0', transform=plt.gca().transAxes
                        ##         )
                        idx+=1
                    else:
                        axes[i][j].remove()
                        
            fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=4)
            ## plt.legend(lines, labels, loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),
            ##            bbox_transform = plt.gcf().transFigure)
            
            '''
            label= r'($\varepsilon^{VPSC}$, $\varepsilon^{Predicted}_{RECURSIVE}$)'+\
            r' and ($\eta^{VPSC}$ , $\eta^{Predicted}_{RECURSIVE}$) '+\
            r', in function of $\bar{\varepsilon}$ '+ \
            ' for {0} and IRUN ={1}'.format(str.split(ds_file1, '.')[0], iruned[ig], y=0.1)
            '''
            
            label= r'$C_{{ij}}^{{VPSC/true}}=f(\bar{\varepsilon})$, and '+\
                r'$C_{{ij}}^{{predicted_{{RECURSIVE}}}}=f(\bar{{\varepsilon}})$'\
                ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.i_run,y=0.1)
                ## ' for {0} and IRUN ={1}'.format(str.split(ds_name,'.')[0],self.iruned[ig],y=0.1)  # @1-.
            
            fig.text(0.8,0.1,label,color='r',fontsize=12,
                     horizontalalignment='right',verticalalignment='top',
                     backgroundcolor='1.0'
                     )
            
            plt.show()
            fig.savefig(os.path.join(dir_save,'C_VPSC_true-Pred_REC_NONREC'+
                                     str(self.i_run)+'.png'),format='png',
                        ## str(self.iruned[ig])+'.png'),format='png',  # @1-.
                        dpi=100) # -.
    
    def plot_eps_phi(self,ds_name:str,dir_save:str):
        ''' plot $\varepsilon$ and $\phi$ vs. $\bar{\varepsilon}$ and 
            ($C^{VPSC-true}_{ij}$ vs. $C^{pred-REC_NON-REC}_{ij}$-.
        '''
        ## for ig, df_model in enumerate(self.df_true): # @1 decomment this line to plot more than one IRUN-.
        ## for ig, (dfm,dfmp,dfmpr) in enumerate(zip(self.df_true,self.df_pred,self.df_pred_rec)):
        for ig, dfm in enumerate(self.df):
            ## set seome matplotlib params
            case='_out'
            ## https://stackoverflow.com/questions/5036700/how-can-you-dynamically-create-variables
            locals()['eps_teo_{0}'.format(self.i_run)],\
                locals()['phi_teo_{0}'.format(self.i_run)]=ceac(
                    dfm.loc[:,'c11'f'{case}'],
                    dfm.loc[:,'c13'f'{case}'],dfm.loc[:,'c22'f'{case}'],
                    dfm.loc[:,'c33'f'{case}'],dfm.loc[:,'c44'f'{case}'],
                    dfm.loc[:,'c55'f'{case}'],dfm.loc[:,'c66'f'{case}']
                )
            locals()['eps_pred_{0}'.format(self.i_run)],\
                locals()['phi_pred_{0}'.format(self.i_run)]=ceac(
                    self.df_pred[self.i_run].loc[:,'c11'f'{case}'],
                    self.df_pred[self.i_run].loc[:,'c13'f'{case}'],
                    self.df_pred[self.i_run].loc[:,'c22'f'{case}'],
                    self.df_pred[self.i_run].loc[:,'c33'f'{case}'],
                    self.df_pred[self.i_run].loc[:,'c44'f'{case}'],
                    self.df_pred[self.i_run].loc[:,'c55'f'{case}'],
                    self.df_pred[self.i_run].loc[:,'c66'f'{case}']
                )
            locals()['eps_pred_rec_{0}'.format(self.i_run)],\
                locals()['phi_pred_rec_{0}'.format(self.i_run)]=ceac(
                    self.df_pred_rec[self.i_run].loc[:,'c11'f'{case}'],
                    self.df_pred_rec[self.i_run].loc[:,'c13'f'{case}'],
                    self.df_pred_rec[self.i_run].loc[:,'c22'f'{case}'],
                    self.df_pred_rec[self.i_run].loc[:,'c33'f'{case}'],
                    self.df_pred_rec[self.i_run].loc[:,'c44'f'{case}'],
                    self.df_pred_rec[self.i_run].loc[:,'c55'f'{case}'],
                    self.df_pred_rec[self.i_run].loc[:,'c66'f'{case}']
                )

            ## epsilon & phi-.
            eps_teo=eval('eps_teo_{0}'.format(self.i_run))
            eps_pred=eval('eps_pred_{0}'.format(self.i_run))
            eps_pred_rec=eval('eps_pred_rec_{0}'.format(self.i_run))
            phi_teo=eval('phi_teo_{0}'.format(self.i_run))
            phi_pred=eval('phi_pred_{0}'.format(self.i_run))
            phi_pred_rec=eval('phi_pred_rec_{0}'.format(self.i_run))
            
            m_eps_pred,c_eps_pred,pearR_pred_eps=al(eps_teo,eps_pred)
            m_eps_pred_rec,c_eps_pred_rec,pearR_pred_rec_eps=al(eps_teo,eps_pred_rec)
            m_phi_pred,c_phi_pred,pearR_pred_phi=al(phi_teo,phi_pred)
            m_phi_pred_rec,c_phi_pred_rec,pearR_pred_rec_phi=al(phi_teo,phi_pred_rec)

            fig,axs=plt.subplots(2,2,figsize=(18, 9),sharey='row')
            ## axs=np.ravel(axs) ##     ax=ax.flatten() ## ax=ax.flatten()
            ax=axs.flatten() ## ax= ax.flatten()
            plt.rcParams.update({'font.size': 10})

            ## plot NON_RECURSIVE PREDICTION-.
            label=r"$\varepsilon^{{{0}}}_{{{1}}}$".format('VPSC',self.i_run)
            ax[0].scatter(x=dfm['strain'],y=eps_teo,s=20,facecolors='none',
                          edgecolors='r',alpha=0.5,marker='^',
                          c='blue',label=label)
            label=r"$\varepsilon^{{{0}}}_{{{1}}}$".format('NON-REC',self.i_run)
            ax[0].scatter(x=dfm['strain'],y=eps_pred,s=20,facecolors='none',
                          edgecolors='black',alpha= 1.0,marker='o',
                          label=label)
            label=r"$\varepsilon^{{{0}}}_{{{1}}}$".format('REC',self.i_run)
            ax[0].scatter(x=dfm['strain'],y=eps_pred_rec,s=20,facecolors='none',
                          edgecolors='g',alpha= 1.0,marker='o',
                          label=label)

            ax[0].grid()
            ax[0].legend()
            ## ax.set_title(r'$\varepsilon$')
            ax[0].set_xlabel(r'$\bar\varepsilon$')
            ax[0].set_ylabel(r'$\varepsilon$')
            
            label= r"$\varepsilon^{{{0}}}_{{{1}}}-\varepsilon^{{{2}}}_{{{1}}}$".\
                format('PRED',self.i_run,'VPSC')
            ax[1].scatter(x=eps_teo,y=eps_pred,s=40,facecolors='none',edgecolors='black',
                          alpha=1.0,marker='o',label=label ## c='magenta',
                          ## label=r'$\varepsilon^{PRED}_{IRUN=4_{NonIterative}}-\varepsilon^{TEO}_{IRUN=4}$',
                          ## facecolors='none'
                          # edgecolor='k',
                          # alpha=0.1,
                          # marker='^'
                          # c='blue'
                          )
            label= r"$\varepsilon^{{{0}}}_{{{1}}}-\varepsilon^{{{2}}}_{{{1}}}$".\
                format('PRED-REC',self.i_run,'VPSC')
            ax[1].scatter(x=eps_teo,y=eps_pred_rec,s=40,facecolors='none',edgecolors='m',
                          alpha=1.0,marker='p',label=label ## c='magenta',
                          ## label=r'$\varepsilon^{PRED}_{IRUN=4_{NonIterative}}-\varepsilon^{TEO}_{IRUN=4}$',
                          ## facecolors='none'
                          # edgecolor='k',
                          # alpha=0.1,
                          # marker='^'
                          # c='blue'
                          )

            color='red'
            ax[1].plot(eps_teo,eps_teo*m_eps_pred+c_eps_pred,
                       color=color,
                       label="Fit -- r = %6.4f"%(pearR_pred_eps))
            color='b'
            ax[1].plot(eps_teo,eps_teo*m_eps_pred_rec+c_eps_pred_rec,
                       color=color,
                       label="Fit -- r = %6.4f"%(pearR_pred_rec_eps))

            ax[1].set_xlabel(r'$\varepsilon^{Predicted}$')
            ax[1].set_ylabel(r'$\varepsilon^{VPSC-true}$')
            ax[1].legend(loc=2)
            ax[1].grid()

            ## plot NON_RECURSIVE PREDICTION-.
            label=r"$\phi^{{{0}}}_{{{1}}}$".format('VPSC',self.i_run)
            ax[2].scatter(x=dfm['strain'],y=phi_teo,s=20,facecolors='none',
                          edgecolors='r',alpha=0.5,marker='^',
                          c='blue',label=label)
            label=r"$\phi^{{{0}}}_{{{1}}}$".format('NON-REC',self.i_run)
            ax[2].scatter(x=dfm['strain'],y=phi_pred,s=20,facecolors='none',
                          edgecolors='black',alpha= 1.0,marker='o',
                          label=label)
            label=r"$\phi^{{{0}}}_{{{1}}}$".format('REC',self.i_run)
            ax[2].scatter(x=dfm['strain'],y=phi_pred_rec,s=20,facecolors='none',
                          edgecolors='g',alpha= 1.0,marker='o',
                          label=label)

            ax[2].grid()
            ax[2].legend()
            ## ax.set_title(r'$\varepsilon$')
            ax[2].set_xlabel(r'$\bar\varepsilon$')
            ax[2].set_ylabel(r'$\phi$')

            label= r"$\phi^{{{0}}}_{{{1}}}-\phi^{{{2}}}_{{{1}}}$".\
                format('PRED',self.i_run,'VPSC')
            ax[3].scatter(x=phi_teo,y=phi_pred,s=40,facecolors='none',edgecolors='black',
                          alpha=1.0,marker='o',label=label## c='magenta',
                          ## label=r'$\varepsilon^{PRED}_{IRUN=4_{NonIterative}}-\varepsilon^{TEO}_{IRUN=4}$',
                          ## facecolors='none'
                          # edgecolor='k',
                          # alpha=0.1,
                          # marker='^'
                          # c='blue'
                          )
            label= r"$\phi^{{{0}}}_{{{1}}}-\phi^{{{2}}}_{{{1}}}$".\
                format('PRED-REC',self.i_run,'VPSC')
            ax[3].scatter(x=phi_teo,y=phi_pred_rec,s=40,facecolors='none',edgecolors='m',
                          alpha=1.0,marker='p',label=label## c='magenta',
                          ## label=r'$\varepsilon^{PRED}_{IRUN=4_{NonIterative}}-\varepsilon^{TEO}_{IRUN=4}$',
                          ## facecolors='none'
                          # edgecolor='k',
                          # alpha=0.1,
                          # marker='^'
                          # c='blue'
                          )

            color='red'
            ax[3].plot(phi_teo,phi_teo*m_phi_pred+c_phi_pred,
                       color=color,
                       label="Fit -- r = %6.4f"%(pearR_pred_phi))
            color='blue'
            ax[3].plot(phi_teo,phi_teo*m_phi_pred_rec+c_phi_pred_rec,
                       color=color,
                       label="Fit -- r = %6.4f"%(pearR_pred_rec_phi))
            
            ax[3].set_xlabel(r'$\phi^{Predicted}$')
            ax[3].set_ylabel(r'$\phi^{VPSC-True}$')
            ax[3].legend(loc=2)
            ax[3].grid()

        eps_rmse_pred=eermse(eps_teo,eps_pred)
        eps_rmse_pred_rec=eermse(eps_teo,eps_pred_rec)
        phi_rmse_pred= eermse(phi_teo,phi_pred)
        phi_rmse_pred_rec=eermse(phi_teo,phi_pred_rec)

        label_eps='RMSE_NonRec={0}{1}RMSE_Rec={2}'.\
            format(np.round(eps_rmse_pred,9),'\n',np.round(eps_rmse_pred_rec,9))
        label_phi='RMSE_NonRec={0}{1}RMSE_Rec={2}'.\
            format(np.round(phi_rmse_pred,9),'\n',np.round(phi_rmse_pred_rec,9))
        ax[0].text(0.4, 0.8, label_eps, color='g', fontsize=10,
                   horizontalalignment='right', verticalalignment='top',
                   backgroundcolor='1.0', transform=ax[0].transAxes, # transform=ax[0].gca().transAxes
                   )
        ax[2].text(0.4, 0.8, label_phi, color='g', fontsize=10,
                   horizontalalignment='right', verticalalignment='top',
                   backgroundcolor='1.0', transform= ax[2].transAxes, # transform=ax[0].gca().transAxes
                   )

        '''
        ## https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html
        eq1= (r'$\epsilon=\frac{1/8(C_{11}+C_{33})-1/4C_{13}+1/2C_{55}}{1/2(C_{44}+C_{66})}$')
        ax[0].text(1.4, 1.21, eq1, color='b', fontsize=16,
        horizontalalignment='right', verticalalignment='top',
        backgroundcolor='1.0'
        )
        '''
        
        plt.tight_layout()
        ##ticks,labels=plt.xticks()
        ##plt.xticks(ticks[::50], labels[::50])
        '''
        ##plt.suptitle(f'Validation vs. Predicted values of elaastic anisotropy tensor components '+\
            ##             f'In NON-ITERATIVE APPROACH')
        ##plt.show()
        ## input('ENTER')
        ## exit(1)
        '''
        ## https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html
        eq2= (r'$\phi=\frac{1/2(C_{12}+C_{23})}{3/8(C_{11}+C_{33})+1/4C_{13}+1/2C_{55}-'\
        r'(C_{44}+C_{66})}$')
        ax[1].text(2.25, 1.00, eq2, color='r', fontsize=16,
        horizontalalignment='right', verticalalignment='top',
        backgroundcolor='1.0'
        )
        
        ##plt.suptitle(f'Comparison of '+
        ##             r'$\varepsilon$ and $\phi$ '+
        ##             f'elastic anisotropy coefficients '+
        ##             ## f'for IRUN = {str(df_model)[-1]} (not used in training model)'+
        ##             'for {0} IRUNs predicted with ITERATIVE approach using {1}{2}.'.
        ##             format(None, '\n', self.i_run)
        ##             )
        plt.show()

        fig.savefig(os.path.join(dir_save,'EpsPhi_'+str(self.i_run)+'.png'),format='png',dpi=100)
    
    def set_plot_options_all_C(self,dim):
        ''' set genreal option to plot all C tensor component values-.'''
        ##plt.set_context('paper',rc={'font.size':4,
        ##                            'axes.titlesize':4,
        ##                            'axes.labelsize':4})            
        nr,nc=(6,6) if str.lower(dim)=='2d' else (7,6) if dim=='3d' else (None,None)
        return {'font.size': 6}, gridspec.GridSpec(nr,nc),\
            plt.figure(figsize=(20, 10)),sns.color_palette('mako_r',4)


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
    print('{0}{1}{0}'.format('\n'*2,sf.dir_save,'\n'*2))

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
    ret_val=plotObj.plot_C_VPSC_pred(df_rec_main, ds_file, sf.dir_save, iruned)

