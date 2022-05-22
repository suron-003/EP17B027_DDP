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
from mc import *


@jit(nopython=True)
def lq(lat,tq):
    #values from Songbo Jin, Arnab Sen paper
    #tq - max number of mc steps
    #1/tq - rate
    time = np.arange(-tq+1,tq,1)
    temp = Tc*(1-time/tq)
    s = int(len(lat)) 
    for i in range(0,s**2):
        lat = mc_sweep(lat,2*Tc)
       
    for T in temp:
        lat = mc_step(lat,T)
        
    return lat, T