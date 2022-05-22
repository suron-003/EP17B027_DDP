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
from lat import *
from obs import *

@jit(nopython=True)
def mc_step(lat,temp):
    s = int (len(lat))
    x = random.randint(0,s-1)
    y = random.randint(0,s-1)
    vec = -1*lat[x,y]
    de = cell_energy(lat,vec,x,y) - cell_energy(lat,lat[x,y],x,y)
    if (de < 0):
        lat[x,y] = vec
    else:
        prob = math.exp((-1*de)/temp)
        ran = random.uniform(0,1)
        if (ran < prob):
            lat[x,y] = vec
    
    return lat

#MC_updates
@jit(nopython=True)
def mc_sweep(lat,temp):
    s = int(len(lat))
    for i in range(0,s):
        for j in range(0,s):
            x = random.randint(0,s-1)
            y = random.randint(0,s-1)
            vec = -1*lat[x,y]
            de = cell_energy(lat,vec,x,y) - cell_energy(lat,lat[x,y],x,y)
            if (de < 0):
                lat[x,y] = vec
            else:
                prob = math.exp((-1*de)/temp)
                ran = random.uniform(0,1)
                if (ran < prob):
                    lat[x,y] = vec
    
    return lat 