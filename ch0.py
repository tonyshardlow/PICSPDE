"""
Python translation of the MATLAB
codes for Introduction to Computational Stochastic PDEs

Helper routine to setup plotting.
"""
# load standard set of Python modules
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# Provides Python compatibility between 2 and 3
# Works with futher 0.15.2
# Use the following to load
# exec(open("./icspde_preload.py").read())
# First line below  must  be included directly (don't know why)

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (bytes, dict, int, list, object, range, str,
                      ascii, chr, hex, input, next, oct, open,
                      pow, round, super, filter, map, zip)
from future.builtins.disabled import (apply, cmp, coerce, execfile,
                                      file, long, raw_input, reduce, reload,
                                      unicode, xrange, StandardError)
#
import sys
#
from math import *
# Numpy
import numpy as np
# Pylab for plotting
import matplotlib as mpl
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#
def PlotSetup(xinches=5.875,yinches=3.1):
    
    plt.clf()
    plt.gcf().set_size_inches(xinches,yinches)   
    mpl.rcParams['text.usetex']= False
    # font list available at http://matplotlib.org/users/customizing.html
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['xtick.labelsize']=7
    mpl.rcParams['ytick.labelsize']=7


