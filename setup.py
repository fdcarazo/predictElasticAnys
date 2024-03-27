#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
##
## script to save version and other stuffs (for future package)-.
## python setup.py sdist; pip install pip --upgrade; pip install twine;
## twine upload dist/*
##
## @author: Fernando Diego Carazo (@buenaluna)-.
##
## start_date: 
## last_modify (Fr): Tue Mar 26 22:18:28 CET 2024-.
## last_modify (Arg): -.
##
## ======================================================================= END79

## ======================================================================= INI79
## include modulus-.
from pathlib import Path
from setuptools import setup
## ======================================================================= END79

this_directory=Path(__file__).parent
long_description=(this_directory/'README.md').read_text()

VERSION='0.0.1'
DESCRIPTION='Package to plot and visualize the elastic tensor components of olivine obtained with a ML model.'
PACKAGE_NAME='elasAnysTenCompfOlivine'
AUTHOR='Fernando Diego Carazo'
EMAIL='fernandodcarazo@gmail.com'
GITHUB_URL='https://github.com/fdcarazo'

setup(
    name=PACKAGE_NAME,
    packages=[PACKAGE_NAME],
    version=VERSION,
    license='GPL',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=GITHUB_URL,
    keywords=[],
    install_requires=[ 
        'requests',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Physics, Geophysics, and Mechanical and Material Engineers',
        'Topic :: Material Properties :: Mechanical Properties Prediction',
        'License :: GPL :: GLP License',
        'Programming Language :: Python :: 3.9.6',
    ],
)
