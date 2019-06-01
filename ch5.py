"""
Python translation of the MATLAB
codes for Introduction to Computational Stochastic PDEs

Chapter 5 (ch5.py)

Type

import ch5
ch5.exa_bm()
ch5.exa_bb()
ch5.exa_fbm()

Figures (pdf) are generated for Brownian motion, Brownian bridge,
and fractional Brownian motion
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
import ch0
#
assert(1/2==0.5),'Fractions are not treated as floating-point numbers. Need: from __future__ import division' 
#
def bmotion(t):
    """
    A5.1 Page 186
    """
    X=np.zeros(t.size) # start at 0
    for n in range(1,t.size): # time loop
        dt=t[n] - t[n - 1]
        X[n]=X[n - 1] + sqrt(dt) * np.random.randn()
    return X
#    
def bb(t):
    """
    A5.2 Page 195
    """
    W=bmotion(t)
    X=W - W[-1] * (t - t[0]) / (t[-1] - t[0])
    return X
#
def fbm(t,H):
    """
    A5.3 Page 200
    """
    N=t.size
    C_N=np.zeros((N,N))
    for i in range(0,N): # compute covariance matrix
        for j in range(0,N):
            ti=t[i];             tj=t[j]
            C_N[i,j]=0.5 * (ti ** (2 * H) + tj ** (2 * H)
                            - abs(ti - tj) ** (2 * H))
    S,U=np.linalg.eig(C_N)
    xsi=np.random.randn(N)
    X=np.dot(U,  (S ** 0.5) * xsi))
    return X
#
def exa_bm():
    T=10; N=100;
    t=np.linspace(0,T,N)
    X=bmotion(t)
    plt.figure(1)
    ch0.PlotSetup()
    plt.plot(t,X,'k-')
    plt.xlabel(r'$t$')
    plt.ylabel(r'X')
    plt.savefig('fig5_bm.pdf',bbox_inches='tight')
#
def exa_bb():
    T=10; N=100;
    t=np.linspace(0,T,N)
    X=bb(t)
    plt.figure(1)
    ch0.PlotSetup()
    plt.plot(t,X,'k-')
    plt.xlabel(r'$t$')
    plt.ylabel(r'B')
    plt.savefig('fig5_bb.pdf',bbox_inches='tight')
#
def exa_fbm(H=0.1):
    T=10; N=100;
    t=np.linspace(0,T,N); 
    X=fbm(t,H)
    plt.figure(1)
    ch0.PlotSetup()
    plt.plot(t,X,'k-')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$B^H$')
    plt.savefig('fig5_fbm.pdf',bbox_inches='tight')

