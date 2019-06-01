"""
Python translation of the MATLAB
codes for Introduction to Computational Stochastic PDEs

Chapter 7 (ch7.py)

Type

import ch7
ch7.ex7_31()
ch7.exa7_40()
ch7.exa7_41()
ch7.exa7_42()
ch7.fig7_10n() for n=a,b,c
ch7.fig7_11()
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
fft=np.fft.fft
fft2=np.fft.fft2
ifft=np.fft.ifft
ifft2=np.fft.ifft2
#
import ch0
import ch4
import ch6
#
assert(1/2==0.5),'Fractions are not treated as floating-point numbers. Need: from __future__ import division' 
#

def circ_cov_sample_2d(C_red,n1,n2):
    """
    A7.1 Page 271
    """
    N=n1 * n2
    Lam=N * ifft2(C_red)
    d=np.ravel(np.real(Lam))
    d_minus=np.maximum(- d,0)
    if (np.max(d_minus) > 0):
        print('Invalid covariance rho(D_minus)={x:4.5f}'.format(x=np.max(d_minus)))
    xi=np.random.randn(n1,n2) + 1j*np.random.randn(n1,n2)
    V=(Lam ** 0.5)*xi
    Z=fft2(V) / sqrt(N)
    Z=np.ravel(Z)
    X=np.real(Z)
    Y=np.imag(Z)
    return X,Y


def reduced_cov(n1,n2,dx1,dx2,fhandle):
    """
    A7.2 Page 277
    """
    C_red=np.zeros((2 * n1 - 1,2 * n2 - 1))
    for i in range(2 * n1 - 1):
        for j in range(2 * n2 - 1):
            C_red[i,j]=fhandle((i+1 - n1) * dx1,(j+1 - n2) * dx2)
    return C_red
#
from numba import jit, f8
@jit([f8(f8,f8,f8,f8,f8)])
def gaussA_exp(x1,x2,a11,a22,a12):
    """
    A7.3 Page 278
    """
    c=exp(- ((x1 ** 2 * a11 + x2 ** 2 * a22) - 2 * x1 * x2 * a12))
    return c

def circ_embed_sample_2d(C_red,n1,n2):
    """
    A7.4 Page 279
    """
    N=n1 * n2
    tilde_C_red=np.zeros((2 * n1,2 * n2))
    tilde_C_red[1:2 * n1,1:2 * n2]=C_red
    tilde_C_red=np.fft.fftshift(tilde_C_red)
    u1,u2=circ_cov_sample_2d(tilde_C_red,2 * n1,2 * n2)
    u1=np.ravel(u1)
    u2=np.ravel(u2)
    u1=u1[0:2*n1*n2]
    u1=u1.reshape((n1,2 * n2))
    u1=u1[:,::2]
    u2=u2[0:2*n1*n2]
    u2=u2.reshape((n1,2 * n2))
    u2=u2[:,::2]
    return u1,u2

       
def sep_exp(x1,x2,ell_1,ell_2):
    """
    A7.5 Page 279
    """
    c=exp(- abs(x1) / ell_1 - abs(x2) / ell_2)
    return c

def circ_embed_sample_2dB(C_red,n1,n2,m1,m2):
    """
    A7.6 Page 282
    """
    nn1=n1 + m1
    nn2=n2 + m2
    N=nn1 * nn2
    tilde_C_red=np.zeros((2 * nn1,2 * nn2))
    tilde_C_red[1:2 * nn1,1:2 * nn2]=C_red
    tilde_C_red=np.fft.fftshift(tilde_C_red)
    u1,u2=circ_cov_sample_2d(tilde_C_red,2 * nn1,2 * nn2)
    u1=np.ravel(u1)
    u2=np.ravel(u2)
    u1=u1[0:2 * nn1 * n2]
    u1=u1.reshape((nn1,2 * n2))
    u1=u1[0:n1,::2]
    u2=u2[0:2 * nn1 * n2]
    u2=u2.reshape((nn1,2 * n2))
    u2=u2[0:n1,::2]
    return u1,u2

def turn_band_simple(grid1,grid2):
    """
    A7.7 Page 285
    """
    theta=2 * pi * np.random.uniform()
    e=np.array([cos(theta),sin(theta)])
    xx,yy=np.meshgrid(grid1,grid2)
    tt=np.dot(e,np.vstack([xx.ravel(),yy.ravel()]))
    xi=np.random.randn(2)
    v=sqrt(1 / 2) * np.dot(xi,np.vstack([np.cos(tt),np.sin(tt)]))
    v=v.reshape((grid1.size,grid2.size))
    return v

def turn_band_exp_3d(grid1,grid2,grid3,M,Mpad,ell):
    """
    A7.8 Page 288
    """
    xx,yy,zz=np.meshgrid(grid1,grid2,grid3)
    sum=np.zeros(xx.size)
    T=np.linalg.norm(np.max(np.abs(np.hstack([grid1,grid2,grid3]))))
    gridt=- T + (2 * T / (M - 1)) * np.arange(M + Mpad)
    c=cov_fn(gridt,ell)
    for j in range(M):
        X,Y=ch6.circulant_embed_approx(c)
        e=ch4.uniform_sphere()
        tmp=np.vstack([xx.ravel(),yy.ravel(),zz.ravel()])
        tt=np.dot(e,tmp)
        Xi=np.interp(tt,gridt,X)
        sum=sum + Xi
    v=sum / sqrt(M)
    v=v.reshape((grid1.size,grid2.size,grid3.size))
    return v

def cov_fn(t,ell):
    """
    A7.8 Helper function
    """
    return (1-t/ell)*np.exp(-t/ell)
    
def turn_band_simple2(grid1,grid2,M):
    """
    A7.9 Page 289
    """
    xx,yy=np.meshgrid(grid1,grid2)
    sum=np.zeros(xx.size)
    for j in range(M):
        xi=np.random.randn(2)
        theta=pi * j / M
        e=np.array([cos(theta),sin(theta)])
        tmp=np.vstack([xx.ravel(),yy.ravel()])
        tt=np.dot(e,tmp)
        v=sqrt(1 / 2) * np.dot(xi,np.vstack([np.cos(tt),np.sin(tt)]))
        sum=sum + v
    v=sum / sqrt(M)
    v=v.reshape((grid1.size,grid2.size))
    return v


def turn_band_wm(grid1,grid2,M,q,ell):
    """
    A7.10 Page 291
    """
    xx,yy=np.meshgrid(grid1,grid2)
    sum=np.zeros(xx.size)
    T=np.linalg.norm([np.linalg.norm(grid1,np.inf),
                      np.linalg.norm(grid2,np.inf)])
    for j in range(M):
        theta=j * pi / M
        e=np.array([cos(theta),sin(theta)])
        tmp=np.vstack([xx.ravel(),yy.ravel()])
        tt=np.dot(e,tmp)
        gridt,Z=ch6.squad(2 * T,64,64,lambda s: ff(s,q,ell))
        Xi=np.interp(tt,gridt - T,np.real(Z))
        sum=sum + Xi
    u=sum / sqrt(M)
    u=u.reshape(grid1.size,grid2.size)
    return u
#
@jit([f8(f8,f8,f8)])
def ff(s,q,ell):
    f=gamma(q + 1) / gamma(q) * (ell ** 2 * abs(s)) / (1 + (ell * s) ** 2) ** (q + 1)
    return f
#
def exa7_31():
    C_red=np.array([[1, 0.5, 0.5],[0.2, 0.1,0.2],[0.2,0.2,0.1]])
    X,Y=circ_cov_sample_2d(C_red,3,3)
    print(X,"\n",Y)
def exa7_40(): 
    fhandle=lambda x1,x2:gaussA_exp(x1,x2,1,1,0.5)
    C_red=reduced_cov(3,2,1/2,1,fhandle)
    print(C_red)
def exa7_41():
    fhandle1=lambda x1,x2:sep_exp(x1,x2,1/5,1/10)
    C_red=reduced_cov(201,401,1/200,1/200,fhandle1)
    u1,u2=circ_embed_sample_2d(C_red,201,401)
    plt.figure(1)
    ch0.PlotSetup()
    ax = plt.gca()
    x=np.linspace(0,1,201)
    y=np.linspace(0,2,401)
    #ax.plot_wireframe(T,X,ut,rstride=16,cstride=1000,colors='k')
    CS=ax.contourf(y,x,u1,10,cmap=plt.cm.bone)
    #ax.set_zlabel(r'$u$')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$x$')
    plt.colorbar(CS)
    plt.savefig('fig7_6.pdf',bbox_inches='tight')
 
def exa7_42():
    fhandle=lambda x1,x2:gaussA_exp(x1,x2,10,10,0)
    n1=257; n2=257; m1=0;m2=0; dx1=1/(n1-1); dx2=1/(n2-1)
    C_red=reduced_cov(n1+m1,n2+m2,dx1,dx2,fhandle)
    u1,u2=circ_embed_sample_2dB(C_red,n1,n2,m1,m2)
    plt.figure(1)
    ch0.PlotSetup()
    ax = plt.gca()
    x=np.linspace(0,1,n1)
    y=np.linspace(0,2,n2)
    #ax.plot_wireframe(T,X,ut,rstride=16,cstride=1000,colors='k')
    CS=ax.contourf(y,x,u1,10,cmap=plt.cm.bone)
    #ax.set_zlabel(r'$u$')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$x$')
    plt.colorbar(CS)
    plt.savefig('fig7_.pdf',bbox_inches='tight')
 
def fig7_10a():
    grid=np.linspace(0,10,200)
    u=turn_band_simple(grid,grid)
    plt.figure(1)
    ch0.PlotSetup()
    ax = plt.gca()
    #ax.plot_wireframe(T,X,ut,rstride=16,cstride=1000,colors='k')
    CS=ax.contourf(grid,grid,u,10,cmap=plt.cm.bone)
    #ax.set_zlabel(r'$u$')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$x$')
    plt.colorbar(CS)
    plt.savefig('fig7_10a.pdf',bbox_inches='tight')
 
def fig7_10b():
    grid=np.linspace(0,10,200)
    u=turn_band_simple2(grid,grid,10)
    plt.figure(1)
    ch0.PlotSetup()
    ax = plt.gca()
    #ax.plot_wireframe(T,X,ut,rstride=16,cstride=1000,colors='k')
    CS=ax.contourf(grid,grid,u,10,cmap=plt.cm.bone)
    #ax.set_zlabel(r'$u$')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$x$')
    plt.colorbar(CS)
    plt.savefig('fig7_10b.pdf',bbox_inches='tight')
 
def fig7_10c():
    grid=np.linspace(0,10,200)
    u=turn_band_wm(grid,grid,10,1,1)
    plt.figure(1)
    ch0.PlotSetup()
    ax = plt.gca()
    #ax.plot_wireframe(T,X,ut,rstride=16,cstride=1000,colors='k')
    CS=ax.contourf(grid,grid,u,10,cmap=plt.cm.bone)
    #ax.set_zlabel(r'$u$')
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$x$')
    plt.colorbar(CS)
    plt.savefig('fig7_10c.pdf',bbox_inches='tight')

def fig7_11():
    grid=np.linspace(0,10,200)
    u=turn_band_exp_3d(grid,grid,grid,20,0,1)
