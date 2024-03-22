#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## SCRIPT: used to program-.
##
## @AUTHOR: Fernando Diego Carazo (@buenaluna) -.
##
## start_date: Thu Mar 14 14:15:38 CET 2024-.
## last_modify (Fr): -.
##

## ======================================================================= INI79
## 1- include packages, modules, variables, etc.-.
## print(dir()) # to see the names in the local namespace-.
## ======================================================================= END79

## ======================================================================= INI79
## 2- Classes, Functions, Methods, etc. definitions-.
## to calculate $\bar{\varepsilon}$ and $\eta$ anisitropy coefficients according
## to et al.-.
def calcula_elas_anys_coef(c11, c12, c13, c23, c33, c44, c55, c66):
    '''
    function to calculate anisotropy coefficients
    according to: et al.-.
    '''
    epsilon=( 1./8.*(c11+c33)- 1./4.*c13+ 1./2.*c55 ) / ( 1./2.*(c44+c66))
    eta= (1./2.* (c12+c23))/ (3./8.*(c11+c33)+ 1./4.*c13+ 1./2.*c55- (c44+c66))
    return epsilon, eta

