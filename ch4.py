"""
Python translation of the MATLAB
codes for Introduction to Computational Stochastic PDEs

Chapter 4 (ch4.py)

Type

import ch4
ch4.exa4() # runs a bunch of routines
ch4.exa4_72() 
ch4.exa4_73()

"""
# load standard set of Python modules
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import sys
if sys.version_info < (3,):
    try:
        from builtins import (bytes, dict, int, list, object, range, str,
                              ascii, chr, hex, input, next, oct, open,
                              pow, round, super, filter, map, zip)
        from future.builtins.disabled import (apply, cmp, coerce, execfile,
                                              file, long, raw_input,
                                              reduce, reload,
                                              unicode, xrange, StandardError)
    except:
        print("need future module")

#
from math import *
# Numpy
import numpy as np
from numpy import matlib
# Symbolic computing module
import sympy as sp
# Scipy
import scipy
from scipy import optimize
from scipy import sparse
from scipy import special
from scipy.sparse import linalg
from scipy import fftpack
# Pylab for plotting
import matplotlib as mpl
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#
import ch3
assert(1/2==0.5),'Fractions are not treated as floating-point numbers. Need: from __future__ import division' 
#
def est_mean_var_func(mu,sigma,M):
    """
    A4.1 Page 163
    """
    X=np.random.randn(M)
    X=mu + sigma * X
    mu_M=np.mean(X)
    sig_sq_M=np.var(X)
    return mu_M,sig_sq_M

def setseed0(M):
    """
    A4.2 Page 165
    """
    s0=np.random.RandomState()
    s0_state=s0.get_state()
    r0=s0.randn(M)
    s0.set_state(s0_state)
    r00=s0.randn(M)
    return r0,r00
    
def setseed1(M):
    """
    A4.3 Page 165
    """
    s1=np.random.RandomState()
    s0=np.random.RandomState()
    r1=s1.randn(M)
    r0=s0.randn(M)
    tmp=np.vstack((r1,
                   r0))
    return  np.cov(tmp)
    
 def parseed():
    """
    A4.4 Page 166
    """
    N=5
    M=6
    sr=np.zeros((N,M))
    stream={};    init_state={}
    for j in range(N):
        stream[j]=np.random.RandomState() # create new rng
        init_state[j]=stream[j].get_state() # save initial state
        sr[j,:]=stream[j].randn(1,M)
#
def srandn(j): 
        stream[j].set_state(init_state[j])# reset state of jth rng
        return stream[j].randn(1,M)
    #
    sr4=srandn(4) # reproduces the 4th row        
    #
    from ipyparallel import Client
    import os
    rc = Client()   
    view = rc[:]
    # tell clients to work in our directory
    view.apply_sync(os.chdir, os.getcwd())
    ar=view.map_sync(lambda j: srandn(j),range(N))
    return sr,sr4,ar
#       
def uniform_ball():
    """
    A4.5 Page 167
    """
    theta=np.random.uniform()*2*pi; S=np.random.uniform()
    X=sqrt(S)*np.array([cos(theta),sin(theta)])
    return X

def uniform_sphere():
    """
    A4.6 Page 168
    """
    z=-1+2*np.random.uniform()
    theta=2*pi*np.random.uniform()
    r=sqrt(1-z*z)
    X=np.array([r*cos(theta),r*sin(theta),z])
    return X

def reject_uniform():
    """
    A4.7 Page 168
    """
    M=0; X=np.ones(2);
    while np.linalg.norm(X)>1:# reject
        M=M+1; X=np.random.uniform(-1,1,2) # generate sample
    return X,M

def gauss_chol(mu,C):
    """
    A4.8 Page 170
    """
    R= scipy.linalg.cholesky(C, lower=True)
    Z=np.random.randn(mu.size)
    X=mu+np.dot(R.T,Z)
    return X

    
def pop_monte(M,T,Dt,baru0,epsilon):
    """
    A4.9 Page 174
    """
    d=baru0.size
    N=int(T//Dt)
    u=np.zeros((M,d))  
    for j in range(M):
        u0=baru0 + epsilon * np.random.uniform(-1,1,d)
        t,usample=ch3.exp_euler(u0,T,N,d,f_pop)
        u[j,:]=usample[:,-1]
    bar_x,sig95=monte(u[:,0])
    return bar_x,sig95
    
def f_pop(u):
    return np.array([u[0]*(1-u[1]),
                     u[1]*(u[0]-1)])

def monte(samples):
    """
    Helper function for A4.9 Page 174
    """
    M=samples.size
    conf95=2 * sqrt(np.var(samples) / M)
    sample_av=np.mean(samples)
    return sample_av,conf95

    
def pop_monte_anti(M,T,Dt,baru0,epsilon):
    d=baru0.size
    N=int(T//Dt)
    u=np.zeros((2*M,d))  
    for j in range(M):
        u0=baru0 + epsilon * np.random.uniform(-1,1,d)
        t,usample=ch3.exp_euler(u0,T,N,d,f_pop)
        u[j,:]=usample[:,-1]
        u0=2 * baru0 - u0
        t,usample=ch3.exp_euler(u0,T,N,d,f_pop)
        u[j+M,:]=usample[:,-1]
    bar_x,sig95=monte(u[:,0])
    return bar_x,sig95


def exa4():
    mu=1;    sigma=1; M=40
    mu_M,sig_sq_M=est_mean_var_func(mu,sigma,M)
    print("Sample average", mu_M,
          "sample variance", sig_sq_M,
          "based on ", M, "samples for N(1,1)")
    #
    r0,r00=setseed0(M)
    print("Identical samples:\n",r0,"and\n",r00)
    #
    print("Covariance of samples", setseed1(M))
    #
    print("Uniform sample from unit ball at origin", uniform_ball())
    #
    print("Uniform sample from unit sphere at origin", uniform_sphere())
    #
    [X,M]=reject_uniform()
    print("Uniform sample of unit ball", X, "using", M, "attempts (rejection sampling)" )
    #
    mu=np.array([0,0])
    C=np.array([[2,1],[1,2]])
    print("Multivariate Gaussian sample", gauss_chol(mu,C))
    #
def exa4_72():
    M=100
    T=6
    Dt=T/(10*sqrt(M))
    baru0=np.array([0.5,2])
    epsilon=0.2
    bar_x,sig95=pop_monte(M,T,Dt,baru0,epsilon)
    print("mean",bar_x,"sd",sig95)
 

def exa4_73():
    M=100
    T=6
    Dt=T/(10*sqrt(M))
    baru0=np.array([0.5,2])
    epsilon=0.2
    bar_x,sig95=pop_monte_anti(M,T,Dt,baru0,epsilon)
    print("mean",bar_x,"sd",sig95)
 
