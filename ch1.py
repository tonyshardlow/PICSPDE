"""
Python translation of the MATLAB
codes for Introduction to Computational Stochastic PDEs

Chapter 1 (ch1.py)

Type

import ch1
ch1.example()

to generate Fig 1.6 and test Algs 1.1-1.2.
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
from timeit import default_timer as timer
# Fourier transform
fft=np.fft.fft
def icspde_dst1(u):
    return scipy.fftpack.dst(u,type=1,axis=0)/2

# 
import ch0
#
assert(1/2==0.5),'Fractions are not treated as floating-point numbers. Need: from __future__ import division' 
#

def get_coeffs(u,a,b):
    """ 
    A1.1 Page 32. 
    """
    J=u.size - 1;    h=(b - a) / J
    u1=np.hstack(( (u[0] + u[-1 ]) / 2,    u[1:-1]) )
    Uk=(h / (b - a))*np.exp( (- 2 * pi * 1j * a / (b - a) )
                             * np.arange(J ) )*fft(u1)
    assert (J % 2 == 0) # J must be even
    nu= ((2. * pi / (b - a))
         * np.hstack((np.arange(1+J/ 2), np.arange(- J / 2 + 1,0)))) 
    return Uk, nu
    
def get_norm(fhandle,a,b,J):
    """
    A1.2. Page 33
    """
    grid=np.linspace(a,b, J+1)  
    u=fhandle(grid)
    Uk,nu=get_coeffs(u,a,b)
    l2_norm=sqrt(b - a) * np.linalg.norm(Uk)
    dUk=nu*Uk # element-wise multiplication
    h1_norm=np.linalg.norm([l2_norm,
                            sqrt(b - a) * np.linalg.norm(dUk)])
    return l2_norm, h1_norm

# test functions

def f_u1(x):
    return (x*(x-1))**2

def f_u2(x):
    return np.cos(x*pi/2)

# symbolic evaluation of norms for test functions
def get_symbolic(u,x,a,b):
    int1=sp.Integral(u(x)*u(x),(x,a,b))
    du=lambda x:sp.diff(u(x),x)
    int2=sp.Integral(du(x)*du(x),(x,a,b))
    l2_norm=sqrt(int1.evalf())
    h1_semi_norm=sqrt(int2.evalf())
    h1_norm=np.linalg.norm((l2_norm,h1_semi_norm))
    return l2_norm,h1_norm
    
def sp_u1(a,b):
    x=sp.symbols('x')
    u=lambda x:((x*(x-1))**2)
    l2_norm,h1_norm=get_symbolic(u,x,a,b)
    return l2_norm,h1_norm

def sp_u2(a,b):
    x=sp.symbols('x')
    u=lambda x:(sp.cos(x*pi/2))
    l2_norm,h1_norm=get_symbolic(u,x,a,b)
    return l2_norm,h1_norm


# Two examples: compute norms by Fourier transform and symbolically.
# Compare results

def example_u1(a=0,b=1,J=10):
    print("=========\nExample u1 from page 33")
    print("J=",J," [a,b]=[",a,b,"]")
    [l2_norm,h1_norm]=get_norm(f_u1,a,b,J)
    print("Numerically computed")
    print(" L2_norm ",l2_norm,"\n",
          "H1_norm=",h1_norm)
    [l2_norm_sym, h1_norm_sym]=sp_u1(a,b)
    print("Errors for symbolic vs numerical")
    print(" L2_norm ", fabs(l2_norm-l2_norm_sym),"\n",
          "H1_norm=", fabs(h1_norm-h1_norm_sym))
    
def example_u2(a=0,b=1,J=10):
    print("========\nExample u2 from page 33 (H1 norm approximation fails)")
    print("J=",J," [a,b]=[",a,b,"]")
    [l2_norm,h1_norm]=get_norm(f_u2,a,b,J)
    print("Numerically computed")
    print(" L2_norm ",l2_norm,"\n",
          "H1_norm=",h1_norm)
    [l2_norm_sym, h1_norm_sym]=sp_u2(a,b)
    print("Errors for symbolic vs numerical")
    print(" L2_norm ", fabs(l2_norm-l2_norm_sym),"\n",
          "H1_norm=", fabs(h1_norm-h1_norm_sym))

# run both examples    
def example():
    example_u1()
    example_u2()
    fig1_6();
    plt.savefig('fig1_6.pdf',bbox_inches='tight')

# Figure
def fig1_6():
    N=10
    J=np.zeros(N);
    l2_norm_1=np.copy(J);    l2_norm_2=np.copy(J)
    h1_norm_1=np.copy(J);    h1_norm_2=np.copy(J)
    a=0;    b=1; # domain
    for j in range(10):
        N=int(pow(2,(j+3))); J[j]=N; 
        l2_norm_1[j],h1_norm_1[j]=get_norm(f_u1,a,b,N)
    for j in range(10):
        N=int(pow(2,(j+3))); 
        l2_norm_2[j],h1_norm_2[j]=get_norm(f_u2,a,b,N)
    # get symbolic values
    [l2_norm_sym_1, h1_norm_sym_1]=sp_u1(a,b)
    [l2_norm_sym_2, h1_norm_sym_2]=sp_u2(a,b)
         
    # computer errors
    el2_1=abs(l2_norm_1-l2_norm_sym_1)
    ehs_1=abs(h1_norm_1-h1_norm_sym_1)
    el2_2=abs(l2_norm_2-l2_norm_sym_2)
    ehs_2=abs(h1_norm_2-h1_norm_sym_2)
    # plotting
    plt.figure(1)
    ch0.PlotSetup()
    plt.axis('equal')
    plt.subplot(1,2,1)
    #
    plt.loglog(J,el2_1,'k-')
    plt.loglog(J,ehs_1,'k-.')
    plt.xlabel(r'$J$')
    plt.ylabel(r'error')
    plt.title(r'(a)')
    #
    plt.subplot(1,2,2)
    plt.loglog(J,el2_2,'k-')
    plt.loglog(J,ehs_2,'k-.')
    plt.xlabel(r'$J$')
    plt.title(r'(b)')
    #
    plt.tight_layout()    
