"""
Python translation of the MATLAB
codes for Introduction to Computational Stochastic PDEs

Chapter 6 (ch6.py)

Type

import ch6
ch6.fig6_3()
ch6.fig6_4()
ch6.fig6_5()
ch6.fig6_6()
ch6.fig6_7()
ch6.exa6_62()
ch6.exa6_64()
ch6.fig6_9()
ch6.exa6_59()
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
fft=np.fft.fft
fft2=np.fft.fft2
ifft=np.fft.ifft
#
import ch0
import ch1
#
assert(1/2==0.5),'Fractions are not treated as floating-point numbers. Need: from __future__ import division' 
#
def spectral_density(X,T):
    """
    A6.1 Page 222
    """
    Uk,nu=ch1.get_coeffs(X,0,T)
    f=(np.abs(Uk) ** 2) * T / (2 * pi)
    return f,nu
#
def fkl_1d(J,T,ell):
    """
    A6.2 Page 224
    """
    kk=np.arange(0,2 * J ) # range of k
    b=np.sqrt(1 / T * np.exp(- pi * (kk * ell / T) ** 2))
    b[0]=sqrt(0.5) * b[0]
    xi=np.random.uniform(0,1,2 * J) * sqrt(12) - sqrt(3)
    XJ=np.real(fft(b*xi))
    XJ=XJ[0:J]
    return XJ
#    
def quad_sinc(t,J,ell):
    """
    A6.3 Page 235
    """
    R=pi / ell
    nustep=2 * R / J
    Z=(np.exp(- 1j * t * R) * np.dot(np.random.randn(2), [1j,1]) / sqrt(2)
       +np.sum(np.exp(1j * t * (- R + j * nustep)) * np.dot(np.random.randn(2),[1j,1]) for j in range(J-2))
       + np.exp(1j * t * R) * np.dot(np.random.randn(2) , [1j,1]) / sqrt(2))
    # Z=Z * sqrt(ell / (2 * pi)) misprint
    Z=Z * sqrt(1 / (J-1)) # correction 7-Aug 2024
    return Z

    
def squad(T,N,M,fhandle):
    """
    A6.4 Page 239
    """
    dt=T / (N - 1);    t=np.linspace(0,T,N)
    R=pi / dt;    dnu=2 * pi / (N * dt * M)
    Z=np.zeros(N);    coeff=np.zeros(N,dtype='complex128')
    for m in range(M):
        for k in range(N):
            nu=- R + ((k - 1) * M + (m - 1)) * dnu
            xi=np.dot(np.random.randn(2),[1,1j])
            coeff[k]=sqrt(fhandle(nu) * dnu) * xi
            if ((m == 1 and k == 1) or (m == M and k == N)):
                coeff[k]=coeff[k] / sqrt(2)
        Zi=N *ifft(coeff)
        Z=Z + np.exp(1j * (- R + (m - 1) * dnu) * t)*Zi
    return t,Z

def interp_quad(s,N,M,fhandle):
    """
    A6.6 Page 240
    """
    T=np.max(s) - np.min(s)
    t,Z=squad(T,N,M,fhandle)
    Zr=np.real(Z)
    X=np.interp(s,t + np.min(s),Zr)
    Zi=np.imag(Z)
    Y=np.interp(s,t + np.min(s),Zi)
    return X,Y

def quad_wm(s,N,M,q):
    """
    A6.7 Page 241
    """
    X,Y=interp_quad(s,N,M,lambda nu: f_wm(nu,q))
    return X,Y

def f_wm(nu,q):
    """
    A6.7 helper function that gives WM spectral density
    """        
    const=gamma(q + 0.5) / (gamma(q) * gamma(0.5))
    f=const / ((1 + nu * nu) ** (q + 0.5))
    return f


def circ_cov_sample(c):
    """
    A6.7 Page 246
    """
    N=c.size
    d=ifft(c) * N
    xi=np.dot(np.random.randn(N,2), [1,1j])
    Z=fft(np.multiply(d ** 0.5,xi)) / sqrt(N)
    X=np.real(Z)
    Y=np.imag(Z)
    return X,Y

def circulant_embed_sample(c):
    """
    A6.8 Page 247
    """
    # create first column of C_tilde
    tilde_c=np.hstack([c,c[-2:0:-1]])
    # obtain 2 samples from N(0,C_tilde)
    X,Y=circ_cov_sample(tilde_c)
    # extract samples from N(0,C)
    N=c.size;    X=X[0:N];    Y=Y[0:N]
    return X,Y

def circulant_exp(N,dt,ell):
    """
    A6.9 Page 248
    """
    t=np.arange(N)*dt; 
    c=np.exp(- np.abs(t) / ell)
    X,Y=circulant_embed_sample(c)
    return t,X,Y# deleted last return value c

def circulant_embed_approx(c):
    """
    A6.10 Page 251
    """
    tilde_c=np.hstack([c,c[-2:0:-1]])
    tilde_N=tilde_c.size
    d=np.real(ifft(tilde_c)) * tilde_N
    d_minus=np.maximum(- d,0)
    d_pos=np.maximum(d,0)
    if (np.max(d_minus) > 0):
        print('rho(D_minus)={x:0.5g}'.format(x=np.max(d_minus)))
    xi=np.dot(np.random.randn(tilde_N,2), [1,1j])
    Z=fft(np.multiply(d_pos ** 0.5,xi)) / sqrt(tilde_N)
    N=c.size;    X=np.real(Z[0:N]);    Y=np.imag(Z[0:N])
    return X,Y


def circulant_wm(N,M,dt,q):
    """
    A6.11 Page 252
    """
    Ndash=N + M - 1
    c=np.zeros(Ndash + 1)
    T=(Ndash+1)*dt;    t=np.linspace(0,T,Ndash+1)
    c[0]=1 # t=0 is special, due to singularity in Bessel fn
    const=2 ** (q - 1) * gamma(q)
    for i in range(1,Ndash + 1):
        c[i]=(t[i] ** q) * scipy.special.kv(q,t[i]) / const
    X,Y=circulant_embed_approx(c)
    X=X[0:N];    Y=Y[0:N];    t=t[0:N]
    return t,X,Y,c
#
def fig6_3():
    noSamples=10
    J=3200;    dt=1/J
    fc=np.zeros(J+2)
    
    for m in range(noSamples):
        sq_rt_lambda=sqrt(2);
        xi=np.random.randn(J+1)
        coeffs=sq_rt_lambda*xi
        y2=np.hstack([0,icspde_dst1(coeffs),0]);
        f,nu=spectral_density(y2,1)
        fc=fc+f
    fc=fc/noSamples
    poly=0*fc+1/(2*pi)
    plt.figure(1)
    ch0.PlotSetup()
    plt.loglog(nu,f,'b.')
    plt.loglog(nu,fc,'k-')
    plt.loglog(nu,poly,'k-.')
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$f(\nu)$')
    plt.savefig('fig6_3.pdf',bbox_inches='tight')


    #
def fig6_4():
    noSamples=10
    J=3200;    dt=1/J
    fc=np.zeros(J)
    for m in range(noSamples):
        t,X,Y=circulant_exp(J+1,dt,1)
        print(X.shape)
        f,nu=spectral_density(X,1)
        fc=fc+f
    fc=fc/noSamples
    poly=(1/pi)/(1+nu**2)
    plt.figure(1)
    ch0.PlotSetup()
    plt.loglog(nu,f,'b.')
    plt.loglog(nu,fc,'k-')
    plt.loglog(nu,poly,'k-.')
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$f(\nu)$')
    plt.savefig('fig6_4.pdf',bbox_inches='tight')

    #
def fig6_5():
    # todo :-(
    J=100
    T=1
    ell=0.1
    XJ=fkl_1d(J,T,ell)
    print(XJ)

def fig6_7():
    t=np.linspace(0,20,200)
    Z=quad_sinc(t,100,2)
    plt.figure(1)
    ch0.PlotSetup()
    plt.plot(t,np.real(Z))
    plt.plot(t,np.imag(Z))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$Z$')
    plt.savefig('fig6_7b.pdf',bbox_inches='tight')
#
    
def fig6_6():
    T=30
    N=2**9
    t=np.linspace(0,T,N)
    M=N/4
    vector_q=[0.5,1,1.5,2]
    ch0.PlotSetup()
    for i in range(1,5):
        q=vector_q[i-1]
        X,Y=quad_wm(t,N,M,q)
        plt.subplot(2,2,i)
        plt.plot(t,X)
        plt.plot(t,Y)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$X$')    
    plt.savefig('fig6_6.pdf',bbox_inches='tight')
#
def exa6_62():
    [x,y]=circ_cov_sample(np.array([3,2,1,2]))
    print("x=",x,"\n y=",y)
#
def exa6_64():
    [x,y]=circulant_embed_sample(np.array([5,2,3,4]))
    print("x=",x,"\n y=",y)
def fig6_9():
    N=int(1e3)
    dt=1/(N-1)
    t,X,Y=circulant_exp(N,dt,1)
    ch0.PlotSetup()
    plt.plot(t,X)
    plt.plot(t,Y)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$X$')
    plt.savefig('fig6_9.pdf',bbox_inches='tight')
    #
def exa6_59():
    N=100; M=9900; dt=1/(N-1)
    t,X,Y,c=circulant_wm(N,M,dt,3)
    
