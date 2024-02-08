#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## class to predict using recursive approach-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Tue Feb  6 16:10:28 CET 2024-.
## last_modify (Fr): Tue Feb  6 20:21:59 CET 2024-.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
import pandas as pd
import time
import numpy as np
## for PyTorch
import torch
## from torch import tensor as tt, float32 as tf32


## 2ignore warnings messages-.
import warnings
warnings.filterwarnings('ignore')

class Predict():
    '''
         A class to predict using recursive approach-.-.
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
    def __init__(self, df, model, sca_feat, sca_targ, ir):
        self.df=df
        self.model=model
        self.sca_feat=sca_feat
        self.sca_targ=sca_targ
        self.irun=ir
        self.L=[var for var in df.columns if 'vgrad' in var]
        self.C_in=[var for var in df.columns if '_in' in var]
        self.C_out=[var for var in df.columns if '_out' in var]
        
    def pred_recursive_main(self, n_cases:int):
        ''' method to predict $C^{ijkl}$ recursive'''
        ## df_vpsc= dict() # 2BeDo. To save {key:df_name, val:df}-.
        df_vpsc, df_rec, iruned= list(), list(), list()
        for i in range(n_cases):
            if self.irun == 'random':
                irun=np.random.randint(self.df['irun'].min(), self.df['irun'].max())
                iruned.append(irun)
                df_vpsc.append(self.df[self.df['irun']==irun])
            else:
                iruned.append(self.irun)
                df_vpsc.append(self.df[self.df['irun']==self.irun])

        feat_lab=self.L+self.C_in
        feat_plus_targ=self.L+self.C_in+self.C_out # features+targets-.

        df=self.df.loc[:,feat_lab].copy()
        
        for ig, df_model in enumerate(df_vpsc):
            X_test, y_test= df_model.loc[:,feat_lab], \
                df_model.loc[:,self.C_out] # this database contains STRAIN column-.

            df_temp=df_model.loc[:,feat_plus_targ].copy()

            ## predict using recursive approach-.
            start_time= time.time()
            ## df_recur_1= self.pred_rec(df_temp, model, sca_feat, sca_targ)
            df_recur_1= self.pred_rec(df_temp)
            end_time= time.time()
            print('Elapsed time in Recursive Prediction of IRUN = {0} = {1}{2}'.
                  format(iruned[ig],'\t',end_time-start_time))

            # remove in_features features/variables from predicted dataset-.
            df_recur_1.drop(columns=df_recur_1.columns.difference(self.C_out), inplace=True) 
            ## convert PandasDataFrame of predicted values 2 numpy array-.
            ## it isn't necessary beacuse df_recur_1 is a <class 'pandas.core.frame.DataFrame'>-.
            y_pred_df= pd.DataFrame(df_recur_1, columns=self.C_out)
            ## print(y_pred_df)
            df_rec.append(df_recur_1)
            
        return df_vpsc, df_rec, iruned
    
    ## 
    def pred_rec(self, dfin):
        ''' predict in a recursive way WITH A WHOLE DF NORMALIZED '''
        dfin=dfin.reset_index(drop=True)
        C_out_train= list(map(lambda st: str.replace(st, '_in', '_out'), self.C_in))

        feat_plus_targ=self.L+self.C_in+self.C_out # features+targets-.
        feat_lab=self.L+self.C_in
        df_iter= pd.DataFrame(columns=feat_plus_targ) ## create an empty dataset-.
        
        for i in range(len(dfin)):
            if i==0:
                L=dfin.loc[i, self.L]
                Cijkl_in=dfin.loc[i,self.C_in]
                X=dfin.loc[i,feat_lab]
            else:  ##
                L=dfin.loc[i,self.L]
                Cijkl_in=df_iter.loc[i-1,self.C_out]
                X=pd.concat([L,Cijkl_in])
            X=X.to_numpy(dtype=float).reshape(1,-1)

            if i==0:
                Cijkl_out_transf=dfin.loc[i,self.C_out].to_numpy()
            else:
                with torch.no_grad():
                    Cijkl_out= self.model(torch.tensor(self.sca_feat.transform(X), dtype= torch.float)).detach().numpy()
                    ## Cijkl_out_transf= sca_targ.inverse_transform(Cijkl_out)
                    Cijkl_out_transf= self.sca_targ.inverse_transform(Cijkl_out)
            
            reg=L.values.tolist()
            reg.extend(Cijkl_in.values.tolist())
            if i==0:
                reg.extend(Cijkl_out_transf.tolist())
            else:
                reg.extend(Cijkl_out_transf.tolist()[0])
            
            ## add new values at the end of a DataFrame created-.
            df_iter.loc[i]= reg
            print('Step of deformation = {0:4d}'.format(i))

        return df_iter # return a DataFrame-.

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
    
