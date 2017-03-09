# Shape_Deconv_galaxies



## Staff

Supervisor: Jean-Luc Starck

Author: Hao Shan

Collaborator: Fred Maurice Ngole Mboula, Samuel Farrens

Thank: Ming Jiang

Year: 2016-2017

Version: 1.0

Email: shanhao@xao.ac.cn;   hao.shan@cea.fr

## Contents

   Introduction
   
   Requirements
   
   Details
   
   Examples

## Introduction

This repository contains several primary codes and a sub-function code written in Python with the scientific target to realize shape preserving PSF deconvolution for galaxy images.

optimization problem is a minimization about:

                 argmin_x (1/2)*|| x-y ||_2^2 + || lamda o Phy(x) ||_1 + (beta/2) * || Sy-Sx ||_2^2

## Requirements

In order to run the code in this repository the following packages are required:

Python v 2.7.11

Numpy v 1.9.2

Scipy v 0.15.1

Matplotlib v 1.4.3

iSap package v 3.1. The starlet transformations additionally requires the mr_transform.cc C++ script from the Sparse2D library in the iSap package v 3.1. These C++ scripts will be need to be compiled in order to run (see iSap Documentation for details).

## Details

method4_singlegalaxy.py - a primary code to calculate the deconvolution result with respect to the quality criterion of ellipticity components. The convergence curve points are also drawn. It can be calculated directly using a calculator such as Spyder v 2.0.

subfun.py - a code of sub-functions which can be easily called during calculation.



## Examples

to appear

### sub-function:  grad_alpha_PSF

    


    


### sub-function: grad_alpha_shape_PSF

  



    
