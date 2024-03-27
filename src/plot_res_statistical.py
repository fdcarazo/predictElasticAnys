#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## to plot results (for statistical analysis)-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Wed Mar 13 19:04:56 CET 2024-.
## last_modify (Fr): Tue Mar 26 18:10:45 CET 2024-.
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

from src.plot_res import PlotPredRes

from utils.elastic_calcs import calcula_elas_anys_coef as ceac
from utils.gen_tools import adjust_line as al, eps_eta_rmse as eermse,\
    calcula_dist_bet_tensors as cdbt

class PlotResWithStatistics(PlotPredRes):
    '''
         A class to plot results for statistical analysis)-.
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
    def __init__(self,df,niruns:int,dim:str,df_true:list,df_pred_rec:list,df_pred:list,iruned:list,
                 feat_var:list,target_vars:list,idx_o:list):
        super().__init__(dim,df_true,df_pred_rec,df_pred,iruned,feat_var,target_vars,idx_o)
        self.df=df # original df-.
        self.niruns=niruns # numnbers of VPSC/IRUNS simulations to be used)-,

    def corr_anisitropyc_coefi(self,ds_name:str,dir_save:str):
        '''
        Correlation between $\varepsilon_{out}^{TEO,PRED}$,
        $\phi_{out}^{TEO,PRED}$, the PREDICTION CAME FROM
        NON-ITERATIVE approach-.
        to study the correlation between $\varepsilon_{out}^{TEO,PRED}$, 
        $\phi_{out}^{TEO,PRED}$,  I plot $\varepsilon_{out}^{TEO,PRED}$, 
        $\varepsilon_{out}^{TEO}$, vs. $\varepsilon_{out}^{PRED}$-.
        $\phi_{out}^{TEO}$, vs. $\phi_{out}^{PRED}$-.
        '''

        nr,nc=2,2
        fig,ax=plt.subplots(nrows=nr,ncols=nc,figsize=(12,12))#,sharey='row')
        plt.rcParams.update({'font.size': 10})
        ## https://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib
        from matplotlib import cm
        start,stop,number_of_lines=0.0,1.0,len(self.df_true)
        cm_subsection=np.linspace(start,stop,number_of_lines)

        ax=ax.flatten()
        colors=[cm.jet(x) for x in cm_subsection]

        eps_rmse_pred_acu,eps_rmse_pred_rec_acu=0.0,0.0
        phi_rmse_pred_acu,phi_rmse_pred_rec_acu=0.0,0.0

        for ig, (i_color,dfm,dfmp,dfmpr) in enumerate(zip(colors,self.df_true,self.df_pred,self.df_pred_rec)):
            ## set seome matplotlib params
            case='_out'
            ## https://stackoverflow.com/questions/5036700/how-can-you-dynamically-create-variables
            locals()['eps_teo_{0}'.format(self.iruned[ig])],\
                locals()['phi_teo_{0}'.format(self.iruned[ig])]=ceac(
                    dfm.loc[:,'c11'f'{case}'],
                    dfm.loc[:,'c13'f'{case}'],dfm.loc[:,'c22'f'{case}'],
                    dfm.loc[:,'c33'f'{case}'],dfm.loc[:,'c44'f'{case}'],
                    dfm.loc[:,'c55'f'{case}'],dfm.loc[:,'c66'f'{case}']
                )
            locals()['eps_pred_{0}'.format(self.iruned[ig])],\
                locals()['phi_pred_{0}'.format(self.iruned[ig])]=ceac(
                    dfmp.loc[:,'c11'f'{case}'],
                    dfmp.loc[:,'c13'f'{case}'],dfmp.loc[:,'c22'f'{case}'],
                    dfmp.loc[:,'c33'f'{case}'],dfmp.loc[:,'c44'f'{case}'],
                    dfmp.loc[:,'c55'f'{case}'],dfmp.loc[:,'c66'f'{case}']
                )
            locals()['eps_pred_rec_{0}'.format(self.iruned[ig])],\
                locals()['phi_pred_rec_{0}'.format(self.iruned[ig])]=ceac(
                    dfmpr.loc[:,'c11'f'{case}'],
                    dfmpr.loc[:,'c13'f'{case}'],dfmpr.loc[:,'c22'f'{case}'],
                    dfmpr.loc[:,'c33'f'{case}'],dfmpr.loc[:,'c44'f'{case}'],
                    dfmpr.loc[:,'c55'f'{case}'],dfmpr.loc[:,'c66'f'{case}']
                )

            ## epsilon & phi-.
            eps_teo=eval('eps_teo_{0}'.format(self.iruned[ig]))
            eps_pred=eval('eps_pred_{0}'.format(self.iruned[ig]))
            eps_pred_rec=eval('eps_pred_rec_{0}'.format(self.iruned[ig]))
            phi_teo=eval('phi_teo_{0}'.format(self.iruned[ig]))
            phi_pred=eval('phi_pred_{0}'.format(self.iruned[ig]))
            phi_pred_rec=eval('phi_pred_rec_{0}'.format(self.iruned[ig]))
            
            m_eps_pred,c_eps_pred,_=al(eps_teo,eps_pred)
            m_eps_pred_rec,c_eps_pred_rec,_=al(eps_teo,eps_pred_rec)
            m_phi_pred,c_phi_pred,_=al(phi_teo,phi_pred)
            m_phi_pred_rec,c_phi_pred_rec,_=al(phi_teo,phi_pred_rec)
        
            ## plot NON_RECURSIVE PREDICTION-.
            label=r'$\varepsilon^{{PRED}}_{IRUN={{{self.iruned[ig]}}}}$'+\
                r'-$\varepsilon^{{TEO}}_{IRUN={{{self.iruned[ig]}}}}$'
            ax[0].scatter(x=eps_teo,y=eps_pred,s=20,facecolors='none',
                          edgecolors=i_color,alpha= 1.0,marker='o',
                          label=label)
        
            label=r'$\phi^{{PRED}}_{IRUN={{{self.iruned[ig]}}}}$'+\
                r'-$\phi^{{TEO}}_{IRUN={{{self.iruned[ig]}}}}$'
            ax[1].scatter(x=phi_teo,y=phi_pred,s=20,facecolors='none',
                          edgecolors=i_color,alpha= 1.0,marker='o',
                          label=label)
            ## plot RECURSIVE PREDICTION-.
            label=r'$\varepsilon^{{REC_PRED}}_{IRUN={{{self.iruned[ig]}}}}$'+\
                r'-$\varepsilon^{{TEO}}_{IRUN={{{self.iruned[ig]}}}}$'
            ax[2].scatter(x=eps_teo,y=eps_pred_rec,s=20,facecolors='none',
                          edgecolors=i_color,alpha= 1.0,marker='o',
                          label=label)
        
            label=r'$\phi^{{REC_PRED}}_{IRUN={{{self.iruned[ig]}}}}$'+\
                r'-$\phi^{{TEO}}_{IRUN={{{self.iruned[ig]}}}}$'
            ax[3].scatter(x=phi_teo,y=phi_pred_rec,s=20,facecolors='none',
                          edgecolors=i_color,alpha= 1.0,marker='o',
                          label=label)

            color=i_color

            eps_rmse_pred_acu += eermse(eps_teo,eps_pred)
            eps_rmse_pred_rec_acu +=eermse(eps_teo,eps_pred_rec)
            phi_rmse_pred_acu += eermse(phi_teo,phi_pred)
            phi_rmse_pred_rec_acu += eermse(phi_teo,phi_pred_rec)

            ## plt.colorbar()

        ax[0].grid();ax[1].grid();ax[2].grid();ax[3].grid()

        ##plt.suptitle(f'Comparison of '+
        ##             r'$\varepsilon$ and $\phi$ '+
        ##             f'elastic anisotropy coefficients '+
        ##             ## f'for IRUN = {str(df_model)[-1]} (not used in training model)'+
        ##             'for {0} IRUNs predicted with NON-ITERATIVE approach using {1}{2}.'.
        ##             format(n_cases, '\n',str.split(ds_file1, '.')[0])
        ##            )


        eps_rmse_pred_acu/=len(self.df_true)
        eps_rmse_pred_rec_acu/=len(self.df_true)
        phi_rmse_pred_acu/=len(self.df_true)
        phi_rmse_pred_rec_acu/=len(self.df_true)
    
        ax[0].set_xlabel(r'$\varepsilon^{VPSC/true}$')
        ax[0].set_ylabel(r'$\varepsilon^{Predicted}_{NON-RECURSIVE}$')
        ax[1].set_xlabel(r'$\phi^{VPSC/true}$')
        ax[1].set_ylabel(r'$\phi^{Predicted}_{NON-RECURSIVE}$')

        ax[2].set_xlabel(r'$\varepsilon^{VPSC/true}$')
        ax[2].set_ylabel(r'$\varepsilon^{Predicted}_{RECURSIVE}$')
        ax[3].set_xlabel(r'$\phi^{VPSC/true}$')
        ax[3].set_ylabel(r'$\phi^{Predicted}_{RECURSIVE}$')

        ## print('RMSE = {0}'.format(np.round(phi_rmse,9)))
        ## print('RMSE = {0}'.format(np.round(eps_rmse,9)))

        label_eps_pred='RMSE (mean) = {0}'.format(np.round(eps_rmse_pred_acu,9))
        label_eps_pred_rec='RMSE (mean) = {0}'.format(np.round(eps_rmse_pred_rec_acu,9))
        label_phi_pred='RMSE (mean)= {0}'.format(np.round(phi_rmse_pred_acu,9))
        label_phi_pred_rec='RMSE (mean)= {0}'.format(np.round(phi_rmse_pred_rec_acu,9))
        
        ax[0].text(0.4,0.8,label_eps_pred,color='g',fontsize=10,
                   horizontalalignment='right',verticalalignment='top',
                   backgroundcolor='1.0',transform=ax[0].transAxes, # transform=ax[0].gca().transAxes
                   )
        ax[1].text(0.4,0.8,label_phi_pred,color='g',fontsize=10,
                   horizontalalignment='right',verticalalignment='top',
                   backgroundcolor='1.0',transform= ax[1].transAxes, # transform=ax[0].gca().transAxes
                   )

        ax[2].text(0.4,0.8,label_eps_pred_rec,color='g',fontsize=10,
                   horizontalalignment='right',verticalalignment='top',
                   backgroundcolor='1.0',transform=ax[2].transAxes, # transform=ax[0].gca().transAxes
                   )
        ax[3].text(0.4,0.8,label_phi_pred_rec,color='g',fontsize=10,
                   horizontalalignment='right',verticalalignment='top',
                   backgroundcolor='1.0',transform= ax[3].transAxes, # transform=ax[0].gca().transAxes
                   )
        plt.show()

        fig.savefig(os.path.join(dir_save,'epsPhi-VPSC_Pred''.png'),
                    format='png',dpi=100)

    def plot_diff_C_comp(self,ds_name:str,dir_save:str):
        '''
        method to calculate the mean and standard deviation of the difference
        beteween $C^{VPSC/True}_{ij}$ and $C^{predicted}_{ij}$-.
        '''

        ## $\Delta C^{ij}$ pandasDataFrame to save the errors for all IRUNs-.
        columns=self.target_var+['strain']
        delta_cij_pred_strain=pd.DataFrame(columns=columns)
        delta_cij_pred_rec_strain=pd.DataFrame(columns=columns)
        
        for idf, df_model in enumerate(self.df_true): # could be self.df_pred or self.df_pred.Prec
            delta_cij_pred=pd.DataFrame(columns=self.target_var)
            delta_cij_pred_rec=pd.DataFrame(columns=self.target_var)

            delta_cij_pred=delta_cij_pred.append(np.round(np.abs(
                self.df_true[idf][self.target_var].values-self.df_pred[idf]),7),
                                                 ignore_index=True)
            delta_cij_pred_rec=delta_cij_pred_rec.append(np.round(np.abs(
                self.df_true[idf][self.target_var].values-self.df_pred_rec[idf]),7),
                                                         ignore_index=True)
            ## add strain column-.
            delta_cij_pred['strain']=df_model.strain.values
            delta_cij_pred_rec['strain']=df_model.strain.values

            ## save IRUN error in general pandasDF-.
            delta_cij_pred_strain=delta_cij_pred_strain.append(delta_cij_pred,ignore_index=True)
            delta_cij_pred_rec_strain=delta_cij_pred_rec_strain.append(delta_cij_pred_rec,ignore_index=True)

        # 2Flat MultiIndex columns-.
        result_pred=delta_cij_pred_strain.groupby('strain').agg([np.mean, np.std]).reset_index()
        result_pred.columns=result_pred.columns.map('_'.join) # 2Flat MultiIndex columns-.
        result_pred_rec=delta_cij_pred_rec_strain.groupby('strain').agg([np.mean, np.std]).reset_index()
        result_pred_rec.columns=result_pred_rec.columns.map('_'.join) # 2Flat MultiIndex columns-.
        
        ## set matplotlib options-.
        plt.rcParams.update({'font.size': 8})
        gs=gridspec.GridSpec(6, 4) # 24 subfigures-.
        fig=plt.figure(figsize=(20, 10))
        palette=sns.color_palette('mako_r',4)
        
        for idx,col in enumerate(self.target_var,start=0):
            ax=fig.add_subplot(gs[idx])
            
            ax.errorbar(result_pred['strain_'],
                        result_pred[f'{col}_mean'],
                        result_pred[f'{col}_std'],
                        fmt='o',
                        label=r'$|\Delta C^{NO-REC}_{ij}|$',
                        marker='s',
                        mfc='red',
                        mec='green',
                        ms=4,
                        mew=1
                        )

            ax.errorbar(result_pred_rec['strain_'],
                        result_pred_rec[f'{col}_mean'],
                        result_pred_rec[f'{col}_std'],
                        fmt='p',
                        label=r'$|\Delta C^{REC}_{ij}|$',
                        marker='p',
                        mfc='blue',
                        mec='cyan',
                        ms=4,
                        mew=1
                        )

            '''
            x= np.arange(0,1,40)
            y_val_std=np.round(y_val_std.to_numpy(dtype=float),3)
            print(y_val_std)
            sns.pointplot(x=x, y=y_val_mean, errorbar=y_val_std, capsize=.4, color='.5', ax=ax)
            '''
            
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
            plt.xlim([0.0,2.0])
            
            fig.legend(*ax.get_legend_handles_labels(),
                       loc='lower center', ncol=4)
        ## plt.legend(lines,labels,loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
        ##            bbox_transform=plt.gcf().transFigure)
        
        ## 2 use \ bar in latexMode-.
        ## https://stackoverflow.com/questions/65702154/
        ## problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        
        label= r'$abs(\Delta C_{{ij}})= |C_{{ij}}^{{VPSC}}-C_{{ij}}^{{PREDICTED}}|= f(\bar{\varepsilon})$'## +\
        ##    ' , for {0}'.format(self.i_run)
        ## r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
            ## r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
            
        fig.text(0.6,0.1,label,color='r',fontsize=16,
                 horizontalalignment='right',verticalalignment='top',
                 backgroundcolor='1.0')
        
        plt.show()
        fig.savefig(os.path.join(dir_save,'DeltaCijkl.png'),format='png',dpi=100)
        
    def plot_diff_C_comp_1(self,ds_name:str,dir_save:str):
        '''
        method to calculate the mean and standard deviation of the difference
        beteween $C^{VPSC/True}_{ij}$ and $C^{predicted}_{ij}$-.
        '''

        ## $\Delta C^{ij}$ pandasDataFrame to save the errors for all IRUNs-.
        columns=self.target_var+['strain']
        delta_cij_pred_strain=pd.DataFrame(columns=columns)
        delta_cij_pred_rec_strain=pd.DataFrame(columns=columns)
        
        for idf, df_model in enumerate(self.df_true): # could be self.df_pred or self.df_pred.Prec
            delta_cij_pred=pd.DataFrame(columns=self.target_var)
            delta_cij_pred_rec=pd.DataFrame(columns=self.target_var)

            delta_cij_pred=pd.concat([delta_cij_pred,np.round(np.abs(
                self.df_true[idf][self.target_var].values-self.df_pred[idf]),2)])
            delta_cij_pred_rec=pd.concat([delta_cij_pred_rec,np.round(np.abs(
                self.df_true[idf][self.target_var].values-self.df_pred_rec[idf]),2)])
            ## add strain column-.
            delta_cij_pred['strain']=df_model.strain.values
            delta_cij_pred_rec['strain']=df_model.strain.values

            ## save IRUN error in general pandasDF-.
            delta_cij_pred_strain=pd.concat([delta_cij_pred_strain,delta_cij_pred])
            delta_cij_pred_rec_strain=pd.concat([delta_cij_pred_rec_strain,delta_cij_pred_rec])

        # 2Flat MultiIndex columns-.
        result_pred=delta_cij_pred_strain.groupby('strain').agg([np.mean, np.std]).reset_index()
        result_pred.columns=result_pred.columns.map('_'.join) # 2Flat MultiIndex columns-.
        result_pred_rec=delta_cij_pred_rec_strain.groupby('strain').agg([np.mean, np.std]).reset_index()
        result_pred_rec.columns=result_pred_rec.columns.map('_'.join) # 2Flat MultiIndex columns-.
        
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
                    
                    ax.errorbar(result_pred['strain_'],
                                result_pred[f'{col}_mean'],
                                result_pred[f'{col}_std'],
                                fmt='o',
                                label=r'$|\Delta C^{NO-REC}_{ij}|$',
                                marker='s',
                                mfc='red',
                                mec='green',
                                ms=4,
                                mew=1
                                )
                    
                    ax.errorbar(result_pred_rec['strain_'],
                                result_pred_rec[f'{col}_mean'],
                                result_pred_rec[f'{col}_std'],
                                fmt='p',
                                label=r'$|\Delta C^{REC}_{ij}|$',
                                marker='p',
                                mfc='blue',
                                mec='cyan',
                                ms=4,
                                mew=1
                                )

                    '''
                    x= np.arange(0,1,40)
                    y_val_std=np.round(y_val_std.to_numpy(dtype=float),3)
                    print(y_val_std)
                    sns.pointplot(x=x, y=y_val_mean, errorbar=y_val_std, capsize=.4, color='.5', ax=ax)
                    '''
            
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
                    plt.xlim([0.0,result_pred['strain_'].max()])
                    
                    idx+=1
                        
                else:
                    axes[i][j].remove()
                    
        fig.legend(*ax.get_legend_handles_labels(),
                   loc='lower center', ncol=4)
        ## plt.legend(lines,labels,loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),
        ##            bbox_transform=plt.gcf().transFigure)
        
        ## 2 use \ bar in latexMode-.
        ## https://stackoverflow.com/questions/65702154/
        ## problem-with-latex-format-string-in-python-plot-with-double-subscripts-and-expec
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
        
        label= r'$abs(\Delta C_{{ij}})= |C_{{ij}}^{{VPSC}}-C_{{ij}}^{{PREDICTED}}|= f(\bar{\varepsilon})$'## +\
        ##    ' , for {0}'.format(self.i_run)
        ## r' $C_{{ij}}^{{Theoretical-VPSC}}=f(\bar{\varepsilon})$, and '+\
            ## r'$C_{{ij}}^{{Predicted_{{ITERATIVE}}}}=f(\bar{{\varepsilon}})$'\
            
        fig.text(0.6,0.1,label,color='r',fontsize=16,
                 horizontalalignment='right',verticalalignment='top',
                 backgroundcolor='1.0')
        
        plt.show()
        fig.savefig(os.path.join(dir_save,'DeltaCijkl.png'),format='png',dpi=100)
        
    def plot_diff_eps_phi(self,ds_name:str,dir_save:str):
        '''
        method to calculate the mean and standard deviation of the difference
        beteween $(\varepsilon,\phi)^{VPSC/True}_{ij}$ and
        $(\varepsilon,\phi)^{predicted}_{ij}$-.
        '''
        
        ## 
        eps_col=['deltaEps']; columns_eps=eps_col+['strain']
        delta_eps_pred_strain=pd.DataFrame(columns=columns_eps)
        delta_eps_pred_rec_strain=pd.DataFrame(columns=columns_eps)
        phi_col=['deltaPhi']; columns_phi=eps_col+['strain']
        delta_phi_pred_strain=pd.DataFrame(columns=columns_phi)
        delta_phi_pred_rec_strain=pd.DataFrame(columns=columns_phi)

        for ig, (dfm,dfmp,dfmpr) in enumerate(zip(self.df_true,self.df_pred,self.df_pred_rec)):
            ## set seome matplotlib params
            case='_out'
            ## https://stackoverflow.com/questions/5036700/how-can-you-dynamically-create-variables
            locals()['eps_teo_{0}'.format(self.iruned[ig])],\
                locals()['phi_teo_{0}'.format(self.iruned[ig])]=ceac(
                    dfm.loc[:,'c11'f'{case}'],
                    dfm.loc[:,'c13'f'{case}'],dfm.loc[:,'c22'f'{case}'],
                    dfm.loc[:,'c33'f'{case}'],dfm.loc[:,'c44'f'{case}'],
                    dfm.loc[:,'c55'f'{case}'],dfm.loc[:,'c66'f'{case}']
                )
            locals()['eps_pred_{0}'.format(self.iruned[ig])],\
                locals()['phi_pred_{0}'.format(self.iruned[ig])]=ceac(
                    dfmp.loc[:,'c11'f'{case}'],
                    dfmp.loc[:,'c13'f'{case}'],dfmp.loc[:,'c22'f'{case}'],
                    dfmp.loc[:,'c33'f'{case}'],dfmp.loc[:,'c44'f'{case}'],
                    dfmp.loc[:,'c55'f'{case}'],dfmp.loc[:,'c66'f'{case}']
                )
            locals()['eps_pred_rec_{0}'.format(self.iruned[ig])],\
                locals()['phi_pred_rec_{0}'.format(self.iruned[ig])]=ceac(
                    dfmpr.loc[:,'c11'f'{case}'],
                    dfmpr.loc[:,'c13'f'{case}'],dfmpr.loc[:,'c22'f'{case}'],
                    dfmpr.loc[:,'c33'f'{case}'],dfmpr.loc[:,'c44'f'{case}'],
                    dfmpr.loc[:,'c55'f'{case}'],dfmpr.loc[:,'c66'f'{case}']
                )
            
            # <class 'pandas.core.series.Series'> 2 numpr.array
            ## print(eval('eps_teo_{0}'.format(self.iruned[ig])))
            ## print(type(eval('eps_teo_{0}'.format(self.iruned[ig]))))
            ## input(55)
            
            eps_true=eval('eps_teo_{0}'.format(self.iruned[ig])).to_numpy()
            eps_pred=eval('eps_pred_{0}'.format(self.iruned[ig])).to_numpy()
            eps_pred_rec=eval('eps_pred_rec_{0}'.format(self.iruned[ig])).to_numpy()
            phi_true=eval('phi_teo_{0}'.format(self.iruned[ig])).to_numpy()
            phi_pred=eval('phi_pred_{0}'.format(self.iruned[ig])).to_numpy()
            phi_pred_rec=eval('phi_pred_rec_{0}'.format(self.iruned[ig])).to_numpy()

            delta_eps_pred=pd.DataFrame(np.round(np.abs(eps_true-eps_pred),3),columns=eps_col)
            delta_eps_pred_rec=pd.DataFrame(np.round(np.abs(eps_true-eps_pred_rec),3),columns=eps_col)
            delta_phi_pred=pd.DataFrame(np.round(np.abs(phi_true-phi_pred),3),columns=phi_col)
            delta_phi_pred_rec=pd.DataFrame(np.round(np.abs(phi_true-phi_pred_rec),3),columns=phi_col)
            
            ## add strain column-.
            delta_eps_pred['strain']=dfm.loc[:,'strain'].values
            delta_eps_pred_rec['strain']=dfm.loc[:,'strain'].values
            delta_phi_pred['strain']=dfm.loc[:,'strain'].values
            delta_phi_pred_rec['strain']=dfm.loc[:,'strain'].values

            ## save IRUN error in general pandasDF-.
            '''
            delta_eps_pred_strain=delta_eps_pred_strain.append(delta_eps_pred,
                                                               ignore_index=True)
            delta_eps_pred_rec_strain=delta_eps_pred_rec_strain.append(delta_eps_pred_rec,
                                                                       ignore_index=True)
            delta_phi_pred_strain=delta_phi_pred_strain.append(delta_phi_pred,
                                                               ignore_index=True)
            delta_phi_pred_rec_strain=delta_phi_pred_rec_strain.append(delta_phi_pred_rec,
                                                                       ignore_index=True)
            '''
            delta_eps_pred_strain=pd.concat([delta_eps_pred_strain,delta_eps_pred])
            delta_eps_pred_rec_strain=pd.concat([delta_eps_pred_rec_strain,delta_eps_pred_rec])
            delta_phi_pred_strain=pd.concat([delta_phi_pred_strain,delta_phi_pred])
            delta_phi_pred_rec_strain=pd.concat([delta_phi_pred_rec_strain,delta_phi_pred_rec])

        ## 2Flat MultiIndex columns-.
        result_eps_pred=delta_eps_pred_strain.groupby('strain').agg(['mean','std']).reset_index()
        result_eps_pred.columns=result_eps_pred.columns.map('_'.join) # 2Flat MultiIndex columns-.
        result_eps_pred_rec=delta_eps_pred_rec_strain.groupby('strain').agg(['mean','std']).reset_index()
        result_eps_pred_rec.columns=result_eps_pred_rec.columns.map('_'.join) # 2Flat MultiIndex columns-.
        result_phi_pred=delta_phi_pred_strain.groupby('strain').agg([np.mean,np.std]).reset_index()
        result_phi_pred.columns=result_phi_pred.columns.map('_'.join) # 2Flat MultiIndex columns-.
        result_phi_pred_rec=delta_phi_pred_rec_strain.groupby('strain').agg([np.mean,np.std]).reset_index()
        result_phi_pred_rec.columns=result_phi_pred_rec.columns.map('_'.join) # 2Flat MultiIndex columns-.
        
        '''
        ## different forsm to print MultiIndex-.
        for col in result_phi_pred.columns.levels[0]:
        if col=='strain': pass
        else: print(result_phi_pred[col].loc[:,['mean','std']])
        ## print(result_phi.loc[:,(col, ['mean','std'])])
        ########## input('')
        '''
        
        plt.rcParams.update({'font.size': 12})
        fig,ax=plt.subplots(1,2, figsize=(12,6)) #, sharey='row')
        ax=ax.flatten()
        
        ## plot $\phi$ coefficient-.
        ax[0].errorbar(result_phi_pred['strain_'],
                       result_phi_pred['deltaPhi_mean'],
                       result_phi_pred['deltaPhi_std'],
                       fmt='o',
                       label=r'$|\phi^{{VPSC}}-\phi^{{PRED-NON-REC}}|$',
                       marker='s',
                       mfc='red',
                       mec='green',
                       ms=5,
                       mew=1
                       )
        ax[0].errorbar(result_phi_pred_rec['strain_'],
                       result_phi_pred_rec['deltaPhi_mean'],
                       result_phi_pred_rec['deltaPhi_std'],
                       fmt='o',
                       label=r'$|\phi^{{VPSC}}-\phi^{{PRED-REC}}|$',
                       marker='p',
                       mfc='blue',
                       mec='magenta',
                       ms=5,
                       mew=1
                       )
        ## plot $\phi$ coefficient-.
        ax[1].errorbar(result_eps_pred['strain_'],
                       result_eps_pred['deltaEps_mean'],
                       result_eps_pred['deltaEps_std'],
                       fmt='o',
                       label=r'$|\varepsilon^{{VPSC}}-\varepsilon^{{PRED-NON-REC}}|$',
                       marker='s',
                       mfc='red',
                       mec='green',
                       ms=5,
                       mew=1
                       )
        ax[1].errorbar(result_eps_pred_rec['strain_'],
                       result_eps_pred_rec['deltaEps_mean'],
                       result_eps_pred_rec['deltaEps_std'],
                       fmt='o',
                       label=r'$|\varepsilon^{{VPSC}}-\varepsilon^{{PRED-REC}}|$',
                       marker='s',
                       mfc='blue',
                       mec='magenta',
                       ms=5,
                       mew=1
                       )
        
        '''
        plt.suptitle(f'Differences between '+
        r'$\varepsiQlon^{PRED}$ and $\varepsilon^{VPSC}$  '+
        f', and'+
        r'$\phi^{PRED}$ and $\phi^{VPSC}$  '+
        r', in function of $\bar{\varepsilon}$'+
        ' elastic anisotropy coefficients{0}'.format('\n')+
        ## f'for IRUN = {str(df_model)[-1]} (not used in training model)'+
        'for {0} IRUNs '.format(n_cases) + 
        f'predicted with ITERATIVE approach.'
        )
        '''
        
        ax[0].grid(), ax[1].grid()
        ## ax[0].set_xlim(0.0, 1.5)
        ## ax[0].set_xlim(0.0, 1.5)
        ## ax[1].set_xlim(0.0, 1.5)
        ## ax[1].set_xlim(0.0, 1.5)
        
        ax[0].legend(loc='upper left'), ax[1].legend(loc='upper left')
        
        ax[0].set_xlabel(r'$\bar{\varepsilon}$')
        ax[1].set_xlabel(r'$\bar{\varepsilon}$')
        ax[0].set_ylabel(r'$\phi$')
        ax[1].set_ylabel(r'$\varepsilon$')
        
        ## plt.ylabel('{0}'.format(target_var_to_plot))
        ## plt.title('IRUN {0}'.format(iruned))
        plt.show()
        fig.savefig(os.path.join(dir_save,'Phi-Epsilon.png'),format='png',dpi=100)

    def plot_C_dist(self,ds_name:str,dir_save:str):
        '''
        method to calculate the distance between $C^{VPSC/True}_{ij}$
        and $C^{predicted}_{ij}$-.
        '''
        
        ## $||d^{eucl}_{C}||$ pandasDataFrame to save the distances for ALL IRUNs-.
        dist_C_strain_cols=['dist_C']
        strain=['strain']
        dist_C_pred_strain= pd.DataFrame(columns=dist_C_strain_cols+strain)
        dist_C_pred_rec_strain= pd.DataFrame(columns=dist_C_strain_cols+strain)
        
        for ig,(dfm,dfmp,dfmpr) in enumerate(zip(self.df_true,self.df_pred,self.df_pred_rec)):
            
            locals()['dist_tens_pred_{0}'.format(self.iruned[ig])]=cdbt(
                dfm.loc[:,self.target_var].reset_index(drop=True),dfmp)
            locals()['dist_tens_pred_rec_{0}'.format(self.iruned[ig])]=cdbt(
                dfm.loc[:,self.target_var].reset_index(drop=True),dfmpr)
            
            dist_C_pred=pd.DataFrame(eval('dist_tens_pred_{0}'.format(self.iruned[ig])),columns=dist_C_strain_cols)
            dist_C_pred_rec=pd.DataFrame(eval('dist_tens_pred_rec_{0}'.format(self.iruned[ig])),columns=dist_C_strain_cols)
            ## # list(map(lambda x: str.replace(x, '_out',''), out_targets))
            dist_C_pred['strain']=dfm.loc[:,'strain'].values
            dist_C_pred_rec['strain']=dfm.loc[:,'strain'].values 
            ## save IRUN error in general pandasDF-.
            '''
            dist_C_pred_strain=dist_C_pred_strain.append(dist_C_pred,ignore_index=True)
            dist_C_pred_rec_strain=dist_C_pred_rec_strain.append(dist_C_pred_rec,ignore_index=True)
            '''
            dist_C_pred_strain=pd.concat([dist_C_pred_strain,dist_C_pred])
            dist_C_pred_rec_strain=pd.concat([dist_C_pred_rec_strain,dist_C_pred_rec])

        result_dist_C_pred=dist_C_pred_strain.groupby('strain').agg([np.mean, np.std]).reset_index()
        result_dist_C_pred.columns=result_dist_C_pred.columns.map('_'.join) # 2Flat MultiIndex columns-.
        result_dist_C_pred_rec=dist_C_pred_rec_strain.groupby('strain').agg([np.mean, np.std]).reset_index()
        result_dist_C_pred_rec.columns=result_dist_C_pred_rec.columns.map('_'.join) # 2Flat MultiIndex columns-.

        '''
        ## to plot multiIndex
        for col in result_dist_C.columns.levels[0]:
            if col == 'strain': pass
            else: print(result_dist_C[col].loc[:,['mean','std']])
            ## print(result_eta.loc[:,(col, ['mean','std'])])
            ########## input('')
        '''
        
        plt.rcParams.update({'font.size': 10})

        fig, ax= plt.subplots(1,1, figsize=(12,6)) ##, sharey='row')
        ## https://stackoverflow.com/questions/35915431/top-and-bottom-line-on-errorbar-with-python-and-seaborn
        plt.rcParams.update({'font.size': 10, 'errorbar.capsize': 5})
        
        ## plot $\eta$ coefficient-.
        ## (_, caps, _)= plt.errorbar(result_dist_C['strain_'],
        plt.errorbar(result_dist_C_pred['strain_'],
                     result_dist_C_pred['dist_C_mean'],
                     result_dist_C_pred['dist_C_std'],
                     fmt='o',
                     ## label=r'$|d(\mathbf{{C}}^{{VPSC}}-\mathbf{{C}}^{{PRED}})$',
                     label=r'$d_{euclidean}(\mathbb{{C}}^{{VPSC}},\mathbb{{C}}^{{PRED-NON-REC}})$',
                     marker='s',
                     mfc='red',
                     mec='black',
                     ms=5,
                     mew=1,
                     )
        plt.errorbar(result_dist_C_pred_rec['strain_'],
                     result_dist_C_pred_rec['dist_C_mean'],
                     result_dist_C_pred_rec['dist_C_std'],
                     fmt='o',
                     ## label=r'$|d(\mathbf{{C}}^{{VPSC}}-\mathbf{{C}}^{{PRED}})$',
                     label=r'$d_{euclidean}(\mathbb{{C}}^{{VPSC}},\mathbb{{C}}^{{PRED-REC}})$',
                     marker='p',
                     mfc='blue',
                     mec='orange',
                     ms=5,
                     mew=1,
                     )

        '''
        for cap in caps:
        cap.set_markeredgewidth(10)
        cap.set_color('red')
        '''

        '''
        plt.suptitle(f'Euclidean distance between '+
        r'$\mathbb{C}^{PRED}$ and $\mathbb{C}^{VPSC}$  '+
        r'in function of $\bar{\varepsilon}$'+
        ## f'for IRUN = {str(df_model)[-1]} (not used in training model)'+
        '{0}for {1} IRUNs '.format('\n', n_cases) + 
        f'(predictions are obtained using ITERATIVE approach).'
        )
        '''
    
        ax.grid()
        ## ax[0].set_xlim(0.0, 1.5)
        ## ax[0].set_xlim(0.0, 1.5)
        ## ax[1].set_xlim(0.0, 1.5)
        ## ax[1].set_xlim(0.0, 1.5)
        
        ax.legend(loc='upper left')
        plt.xlabel(r'$\bar{\varepsilon}$')
        ## plt.ylabel('{0}'.format(target_var_to_plot))
        plt.ylabel(r'$d_{euclidean}(\mathbb{{C}}^{{VPSC}},\mathbb{{C}}^{{PRED}})$')
        ##plt.title('IRUN {0}'.format(iruned))
        plt.show()
        
        fig.savefig(os.path.join(dir_save,'euclDist-C.png'),format='png',dpi=100)
