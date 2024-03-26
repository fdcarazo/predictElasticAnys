#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## class to predict using non-recursive and recursive approach-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Tue Feb 6 16:10:28 CET 2024-.
## last_modify (Fr): Tue Mar 26 09:00:10 CET 2024-.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
import pandas as pd
import time
import numpy as np
## for PyTorch -.
import torch
## from torch import tensor as tt, float32 as tf32 # if you have problems import FloatTensor()

## 2ignore warnings messages-.
import warnings
warnings.filterwarnings('ignore')

class Predict():
    '''
         A class to predict elastic tensor coefficient using recursive approach-.
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
    def __init__(self,df,model,sca_feat,sca_targ,ir,nnn,feat_list,targ_list):
        self.df=df
        self.model=model
        self.sca_feat=sca_feat
        self.sca_targ=sca_targ
        self.irun=ir
        self.n_nn=nnn # number of neural networks (make sense for BNN)-.
        self.feat_list=feat_list
        self.targ_list=targ_list
        
        self.L=[var for var in self.feat_list if 'vgrad' in var]
        self.C_in=[var for var in self.feat_list if '_in' in var]
        self.C_out=[var for var in self.targ_list if '_out' in var]        
        self.feat_plus_targ=self.feat_list+self.C_out # features+targets-.

    def pred_recursive_main(self,n_cases):
        '''
        method to predict $C^{ijkl}$ using RECURSIVE and NON-RECURSIVE
        approaches-.
        Attributes (2BeCompleted-.)
        ----------
        n_cases: number if VPSC/IRUNS to predict-.
        NOTE - NOTE - NOTE
        ==================
        ** n_cases>1: predict using RERCURSIVE mode (used in statistical analysisis or plots)
                      i.e. src/plot_res_statistical.py -.
        ** n_cases==1: only predict using NON-RERCURSIVE mode (used in RESIDUALS plot)
                       i.e. src/plot_res.py
        ** other n_cases (i.e. 0):
           use one IRUN/VPSC with more than one NN (), used to study the uncertanity
           of the model, i.e. src/bnn_uncertainty.py -.
        Return  (2BeCompleted-.)
        -------
        lists with:
        df_vpsc: df from VPSC/true,
        def_pred: df predicted in non recursive way,
        df_rec: df predicted in recursive way,
        iruned: iruns.
        ====
        '''
        ## df_vpsc= dict() # 2BeDo. To save {key:'df_name' (str), val:df (DataFrame)}-.
        df_vpsc,df_pred,df_rec,iruned=list(),list(),list(),list() # idem [],[],[]
        
        ## if n_cases is not None:
        if n_cases>1:
            for i in range(n_cases):
                if self.irun=='random': irun=np.random.randint(self.df['irun'].min(),self.df['irun'].max())
                iruned.append(irun)
                df_vpsc.append(self.df[self.df['irun']==irun])

            for ig, df_model in enumerate(df_vpsc):
                X_test,y_test=df_model.loc[:,self.feat_list],df_model.loc[:,self.C_out] # not used now-.

                ## ===== 1 ====== predict using RECURSIVE approach-.
                start_time=time.time()
                df_recur_1=self.pred_rec(df_model.loc[:,self.feat_plus_targ]) # call predictive function-.
                self.print_pred_time('RECURSIVE',iruned[ig],abs(start_time-time.time()))

                ## ===== 2 ====== predict using NON-RECURSIVE/CLASSICAL approach-.
                start_time=time.time()
                df_pred_1=pd.DataFrame(np.round(self.predict(df_model.loc[:,self.feat_list]),2),columns=self.C_out)
                self.print_pred_time('NON-RECURSIVE',iruned[ig],abs(start_time-time.time()))
                
                ## remove in_features features/variables in predicted dataset-.
                df_recur_1.drop(columns=df_recur_1.columns.difference(self.C_out),inplace=True)
                ## add $C^{ij}_{out}$ predicted tensor to df_pred and df_rec lists-.
                df_rec.append(df_recur_1); df_pred.append(df_pred_1)
        elif n_cases==1: # only predict using NON-RERCURSIVE mode (used in RESIDUALS plot)-.
            start_time=time.time()
            df_pred_1=pd.DataFrame(np.round(self.predict(self.df.loc[:,self.feat_list]),2),columns=self.C_out)
            self.print_pred_time('NON-RECURSIVE',None,abs(start_time-time.time()))
            df_vpsc.append(self.df); df_pred.append(df_pred_1)
        else: # by default I assume the prediction is for BNN (but the advice is to pass 0 (zero))-.
            ## select VPSC/IRUN case (in this case I only have one df or IRUN/VPSC)-.
            if len(list(self.df['irun'].unique()))>1: # to prevent errors when run one case with strain > 2.0-.
                max_irun=max(self.df['irun'].unique().astype(int))
                i_run=np.random.randint(1,max_irun); df=self.df[self.df['irun']==i_run]
            else:
                ## i_run=self.df['irun'].unique().astype(int); df=self.df[self.df['irun']==i_run]
                ## i_run=self.df['irun'].unique().astype(int) ; input(88) ##; df=self.df[self.df['irun']==i_run]
                i_run=max(self.df['irun'].unique().astype(int))
                print(i_run);input(7); df=self.df[self.df['irun']==i_run];input(7)
                print(type(self.df['irun'].unique().astype(int))); input(88)
                ##print(df); input(88)
            df_vpsc.append(df)
            iruned.append(i_run)
            ## run n NNs-.
            for i_nn, nn in enumerate(range(self.n_nn)):
                ## ===== 1 ====== predict using RECURSIVE approach-.
                start_time=time.time()
                df_recur_1=self.pred_rec(df.loc[:,self.feat_plus_targ]) # call predictive function-.
                self.print_pred_time('RECURSIVE for BNN',None,abs(start_time-time.time()))

                ## ===== 2 ====== predict using NON-RECURSIVE/CLASSICAL approach-.
                start_time=time.time()
                df_pred_1=pd.DataFrame(np.round(self.predict(df.loc[:,self.feat_list]),2),columns=self.C_out)
                self.print_pred_time('NON-RECURSIVE for BNN',None,abs(start_time-time.time()))
                
                ## remove in_features features/variables in predicted dataset-.
                df_recur_1.drop(columns=df_recur_1.columns.difference(self.C_out),inplace=True)
                ## add $C^{ij}_{out}$ predicted tensor to df_pred and df_rec lists-.
                df_rec.append(df_recur_1); df_pred.append(df_pred_1)

        return df_vpsc,df_rec,df_pred,iruned
    
    ## function/method to calculate $C^{ij}_{out}$ using predictive approach-.
    def pred_rec(self,dfin):
        ''' predict using recursive approach WITH A WHOLE DF NORMALIZED '''
        dfin=dfin.reset_index(drop=True) # because in function of irun it has different indexs values-.
        df_iter=pd.DataFrame(columns=self.feat_plus_targ) # create an empty dataset-.

        if len(self.C_in) != len(self.C_out):
            C_in=list(map(lambda st: str.replace(st,'_in','_out'),self.C_in))
        else:
            C_in=self.C_out
     
        for i in range(len(dfin)):
            if i==0:
                L=dfin.loc[i,self.L]
                Cijkl_in=dfin.loc[i,self.C_in]
                X=dfin.loc[i,self.feat_list]
            else: #
                L=dfin.loc[i,self.L]
                ## when don't use all $C^{out}{ij}$ variables, you should specify which
                ## variables sohuld to use (the same that you have in $C^{out}{ij}$ but
                ## replacing 'in' by 'out')-.
                Cijkl_in=df_iter.loc[i-1,C_in]  # @2-.
                ## Cijkl_in=df_iter.loc[i-1,self.C_out] # @2-.
                X=pd.concat([L,Cijkl_in])
            X=X.to_numpy(dtype=float).reshape(1,-1)

            ini_time=time.time()
            if i==0:
                Cijkl_out_transf=dfin.loc[i,self.C_out].to_numpy()
            else:
                with torch.no_grad():
                    Cijkl_out=self.model(torch.tensor(self.sca_feat.transform(X),dtype=torch.float)).detach().numpy()
                    ## Cijkl_out_transf= sca_targ.inverse_transform(Cijkl_out)
                    Cijkl_out_transf=self.sca_targ.inverse_transform(Cijkl_out)
            
            reg=L.values.tolist()
            reg.extend(Cijkl_in.values.tolist())
            if i==0:
                reg.extend(Cijkl_out_transf.tolist())
            else:
                reg.extend(Cijkl_out_transf.tolist()[0])
            
            ## add new values at the end of a DataFrame created-.
            df_iter.loc[i]=reg
            fin_time=time.time()
            print('Step of deformation = {0:4d} | elapsed time = {1:12.8f}'.
                  format(i, (abs(ini_time-fin_time))))

        return df_iter # return a DataFrame-.

    @staticmethod
    def print_pred_time(pred_type,ir:int,elapsed_time:float):
        print('Elapsed time in {0} Prediction of IRUN = {1} = {2}{3}'.
              format(pred_type,ir,'\t',elapsed_time))

    ## 2predict using DL's models in one step-.
    ## @classmethod
    def predict(self,df_feat):
        '''
        Arguments:
        ----------
        df_feat: dataframe with features only-.
        '''
        with torch.no_grad():
            self.model.eval() # if torch.tensor has grad .detach().numpy()
            torch_pred=self.model(torch.tensor(self.sca_feat.transform(df_feat),
                                               dtype=torch.float)).detach().numpy()
            return self.sca_targ.inverse_transform(torch_pred)
    
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
    s_f, s_t, s= ms.load_scalers(r_m_s,'scaler_feat.pkl','scaler_targ.pkl','scaler.pkl')

    '''
    print('{0}{1}{0}'.format('\n'*2, mlmodel,))
    print('{0}'.format(s))
    print('{0}'.format(s_f))
    print('{0}'.format(s_t))
    '''
    
    ## 3-- load ML's model and scalers-.
    pred= Predict(df_main, mlmodel, s_f, s_t)
    ## print(df_main.columns.to_list)
    ## print(pred)
    ##df_vpsc_main, df_rec_main= pred.pred_recursive_main(1, mlmodel,s_f,s_t)
    df_vpsc_main, df_rec_main, iruned= pred.pred_recursive_main(2)
    
