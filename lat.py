import numpy as np
from numpy import random, linspace, cos, pi
import math
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import copy
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d
from plotly import __version__
import cmath
import sys
import time
import numba
from numba import jit
from parameters import *

#All lattice related functions
@jit(nopython=True)
def get_lattice_gs(s):
    s = int(s)
    lat = np.arange(int(s*s)).reshape(s,s)
    spins = np.array([1,-1])
    for i in range (0,s):
        for j in range (0,s):
            lat[i,j] = 1
    return lat

@jit
def get_lattice(s):
    s = int(s)
    lat = np.arange(int(s*s)).reshape(s,s)
    spins = np.array([1,-1])
    for i in range (0,s):
        for j in range (0,s):
            lat[i,j] = random.choice(spins)
            if(np.random.uniform(0,1)<dis_prob):
                lat[i,j] = 0
    return lat

def plotting(lat):
    fig = plt.figure()
    plt.imshow(lat)
    plt.colorbar()
    plt.show()

def quiver_plot(lat):
    fig = plt.figure()
    L = int (len(lat))
    
    x, y = np.meshgrid(np.arange(0, L, 1,dtype=int),
                            np.arange(0, L, 1,dtype=int))
    u = np.array([])
    v = np.array([])
    w = np.array([])
    
    for i in range(0,L):
        for j in range(0,L):
            u = np.append(u,np.cos(lat[i][j]))
            v = np.append(v,np.sin(lat[i][j]))
            
    u = u.reshape(L,L)
    v = v.reshape(L,L)
    
    plt.quiver(x,y,u,v, alpha = 1)
    plt.show()