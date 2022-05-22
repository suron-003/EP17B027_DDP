#observables
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
from numpy import linalg as LA

@jit(nopython=True)
def calcmag(lat):
    s = int(len(lat))
    m = 0
    for i in range(0,s):
        for j in range(0,s):
            m = m + lat[i,j]
    m = m/s**2
    
    return (m)

#calculates cell energy
@jit(nopython=True)
def cell_energy(lat,vec,x,y):
    s = int (len(lat))
    x1 = int((x+1)%s)
    x2 = int((x-1)%s)
    y1 = int((y+1)%s)
    y2 = int((y-1)%s)
    e = (-1*j1*vec*(lat[x1,y]+lat[x2,y]+lat[x,y1]+lat[x,y2])) + (-1*j2*vec*(lat[x1,y1]+lat[x2,y2]+lat[x1,y2]+lat[x2,y1]))
    
    return (e)  

#calculate energy of a lattice
@jit(nopython=True)
def calcenergy_total(lat):
    e0 = 0
    s = int(len(lat))
    for x in range(0,s):
        for y in range(0,s):
            x1 = int((x+1)%s)
            x2 = int((x-1)%s)
            y1 = int((y+1)%s)
            y2 = int((y-1)%s)
            e = (-1*j1*lat[x,y]*(lat[x1,y]+lat[x2,y]+lat[x,y1]+lat[x,y2])) + (-1*j2*lat[x,y]*(lat[x1,y1]+lat[x2,y2]+lat[x1,y2]+lat[x2,y1]))
            e0 = e0 + (e/2)
            
    return (e0/s**2)

@jit(nopython=True)
def op_stripe(lat):
    s = int (len(lat))
    psix = 0
    psiy = 0
    for x in range(0,s):
        for y in range(0,s):
            psix = psix + lat[x,y]*np.power(-1,x)
            psiy = psiy + lat[x,y]*np.power(-1,y)
    
    psix = (1/s**2)*psix
    psiy = (1/s**2)*psiy
    psi = np.array([psix,psiy])
    
    return(psi)


@jit(nopython=True)
def eqtcorr(lat,lind):
    gr = np.zeros((int(s**2*(s**2+1)*0.5),2))
    count = 0
    for i in range(0,len(lind)):
        mind = lind[i:,:]
        for j in range(0,len(mind)):
            dist = np.abs(mind[j,:] - lind[i,:])
            if(dist[0]>s-dist[0]):
                dist[0] = s-dist[0]
            if(dist[1]>s-dist[1]):
                dist[1] = s-dist[1]
            #gr[count,:] = np.array([dist[0]**2+dist[1]**2,lat[lind[i,0],lind[i,1]][0]*lat[mind[j,0],mind[j,1]]][0]+
            # lat[lind[i,0],lind[i,1]][1]*lat[mind[j,0],mind[j,1]]][1])
            gr[count,:] = np.array([dist[0]**2+dist[1]**2,lat[lind[i,0],lind[i,1]]*lat[mind[j,0],mind[j,1]]])
            count = count + 1
            #= np.vstack([gr,(dist[0]**2+dist[1]**2,lat[lind[i,0]][lind[i,1]]*lat[mind[j,0]][mind[j,1]])])
    grn = np.zeros(len(gr))
    for i in range(0,len(gr)):
        grn[i] = gr[i,0]
    grn = list(set(grn))
    grn = np.array(grn)
    grn.sort()
    corr = np.zeros(len(grn))
    for i in range(0,len(grn)):
        #print(grn[i])
        count = 0
        sum = 0
        for j in range(0,len(gr)):
            if(gr[j,0]==grn[i]):
                sum = sum + gr[j,1]
                count = count + 1
        corr[i] = sum/count
    
    return grn,corr

def strop(lat):
    latz = np.zeros((s,s,2))
    for x in range(0,s):
        for y in range(0,s):
            x1 = int((x+1)%s)
            y1 = int((y+1)%s)
            psix = lat[x,y]*np.power(-1,x) + lat[x1,y]*np.power(-1,x1)+ lat[x,y1]*np.power(-1,x) + lat[x1,y1]*np.power(-1,x1)
            psiy = lat[x,y]*np.power(-1,y) + lat[x1,y]*np.power(-1,y)+ lat[x,y1]*np.power(-1,y1) + lat[x1,y1]*np.power(-1,y1)
            latz[x,y] = (1/4)*np.array([psix,psiy])
            latz[x,y] = latz[x,y]/LA.norm(latz)
    
    return latz

def nemop(lat):
    latz = np.zeros((s,s),dtype=np.float32)
    for x in range(0,s):
        for y in range(0,s):
            x1 = int((x+1)%s)
            y1 = int((y+1)%s)
            psix = (1/4)*(lat[x,y]*np.power(-1,x) + lat[x1,y]*np.power(-1,x1)+ lat[x,y1]*np.power(-1,x) + lat[x1,y1]*np.power(-1,x1))
            psiy = (1/4)*(lat[x,y]*np.power(-1,y) + lat[x1,y]*np.power(-1,y)+ lat[x,y1]*np.power(-1,y1) + lat[x1,y1]*np.power(-1,y1))
            latz[x,y] = psix**2 - psiy**2
    
    return latz






            




