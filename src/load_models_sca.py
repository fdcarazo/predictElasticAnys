#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## class to laod ML models and scalers (used to train DL)-.
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Tue Feb  6 11:34:31 CET 2024 -.
## last_modify (Fr): Tue Feb  6 15:59:45 CET 2024-.
##
## ======================================================================= INI79

## print(dir()); input(1)

## 1- IMPORT MODULUS ---.
## Import the required packages/libraries/modules-.
## 1-1- GENERAL MODULES -.
from torch.cuda import is_available as ia
from torch import load as tl
from torch import nn
import pickle, sys

from models.ffnn import MLP

## Pytorch Ensembles(torchensemble) models-.
from torchensemble.utils import io
from torchensemble.fusion import FusionRegressor
from torchensemble.voting import VotingRegressor
from torchensemble.bagging import BaggingRegressor
from torchensemble.gradient_boosting import GradientBoostingRegressor
from torchensemble.snapshot_ensemble import SnapshotEnsembleRegressor

## main class-.
## 2BeMod: -.
class ModelScalersStand():
    '''
         A class to load ML's models and scaler/s-.
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
    def __init__(self):
        pass
    
    def load_model(self,root_mod_sca:str, mfn:str):
        '''
        load ML's model
        Attributes
        ----------
        root_mod_sca: PATH (absolute, from ROOT) to ensemble file model-.
        mfn: name of ensemble file model-.
        '''
        ## check if I've GPU-.
        model_file=root_mod_sca+mfn
        map_location='cpu' if not ia() else lambda storage, loc: storage.cuda()
        
        ## Load DL's model-.
        ## with open(model_file, 'rb') as fm: model= torch.load(fm, map_location=torch.device('cpu')) # @ NotEnsemble-.
        ## print(model_file)
        ## print(str.split(root_mod_sca,'/')[-2])
        ## print(str.split(str.split(root_mod_sca,'/')[-2],'-')); input(88)
        
        if(str.split(str.split(root_mod_sca,'/')[-2],'-')[0]!='Ensemble'):
            with open(model_file, 'rb') as fm: model=tl(fm, map_location=map_location)
        else: ## torchensemble model-.
            model_name=str.split(str.split(root_mod_sca,'/')[-2],'-')[1] # GradientBoostingRegressor
            layers_data=[(2048, nn.ReLU(),None), #
                         (21,None,None)]
            model=eval(model_name)(estimator=MLP(25,21,layers_data),n_estimators=10,cuda=False)
            io.load(model,root_mod_sca,map_location)
        return model

    def load_scalers(self,root_mod_sca:str, sca_feat=None, sca_targ=None, sca=None):
        ''' load scalers/standarizers used 2train the DL's modeling'''
        ## scaler/standarizer absolute PATH-.            
        ## load scaler/standarizer-.
        if sca_feat!=None:
            sca_feat_file= root_mod_sca+sca_feat
            with open(sca_feat_file, 'rb') as fs: sca_feat= pickle.load(fs)
        if sca_targ!=None:
            sca_targ_file= root_mod_sca+sca_targ
            with open(sca_targ_file, 'rb') as fs: sca_targ= pickle.load(fs)
        if sca!=None:
            sca_file= root_mod_sca+sca
            with open(sca_file, 'rb') as fs: sca= pickle.load(fs)
        ## print(scaler_feat, scaler_targ, scaler_targ, sep='\n')
        ## ======================================================================= END79
        
        return sca_feat, sca_targ, sca

if __name__=='__main__':
    sys.path.append('/Users/Fernando/scratch/elasAnys/2testModels/')
    ## create object
    ms=ModelScalersStand()
    r_m_s='/Users/Fernando/temp/models/'  # in a real life it's read in Config class-.
    mfn_name='dlModelWithoutHyperOpt.pt'  # in a real life it's read in Config class-.
    mlmodel=ms.load_model(r_m_s, mfn_name)  # in a real life it's read in Config class-.
    ## print(type(mlmodel)) ## MLP.MPL
    # in a real life it's read in Config class-.
    s, s_f, s_t= ms.load_scalers(r_m_s,'scaler.pkl', 'scaler_feat.pkl', 'scaler_targ.pkl')
    print('{0}{1}{0}'.format('\n'*2, mlmodel,))
    print('{0}'.format(s))
    print('{0}'.format(s_f))
    print('{0}'.format(s_t))
