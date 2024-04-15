#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: used to program-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Thu Mar 14 14:15:38 CET 2024-.
## last_modify (Fr): Tue Mar 26 16:04:35 CET 2024-.
##

## ======================================================================= INI79
## 1- include packages, modules, variables, etc.-.
## print(dir()) # to see the names in the local namespace-.
## ======================================================================= END79

## ======================================================================= INI79
## 2- Classes, Functions, Methods, etc. definitions-.
## to calculate $\phi$ and $\epsilon$ anisotropy coefficients according
## to: JOURNAL OF GEOPHYSICAL RESEARCH,VOL. 91, NO. B1, PAGES 511-520, JANUARY10, 1986
## A SIMPLE METHODFOR INVERTING THE AZIMUTHAL ANISOTROPY OF SURFACE WAVES
def calcula_elas_anys_coef(c11,c13,c22,c33,c44,c55,c66):
    '''
    function to calculate anisotropy coefficients
    according to: JOURNAL OF GEOPHYSICAL RESEARCH,VOL. 91, NO. B1,
    PAGES 511-520, JANUARY10, 1986. A SIMPLE METHODFOR INVERTING THE AZIMUTHAL
    ANISOTROPY OF SURFACE WAVES-.
    '''
    ## eta=(1./8.*(c11+c33)- 1./4.*c13+ 1./2.*c55)/(1./2.*(c44+c66))
    phi=(c22)/(3./8.*(c11+c33)+1./4.*c13+1./2.*c55)
    ## epsilon=(1./2.* (c12+c23))/(3./8.*(c11+c33)+1./4.*c13+1./2.*c55-(c44+c66))
    xi=(1./8.*(c11+c33)- 1./4.*c13+ 1./2.*c55)/(1./2.*(c44+c66))
    return xi, phi
## ======================================================================= END79
