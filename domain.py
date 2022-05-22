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

@jit(nopython=True)
def nz4(lat):#plaquette
    s = int(len(lat))
    latz = np.zeros((s,s))
    for i in range(0,s):
        for j in range(0,s):
            i1 = (i+1)%s
            j1 = (j+1)%s
            tempx = (lat[i,j]*np.power(-1,i)) + (lat[i1,j]*np.power(-1,i1))  
            + (lat[i,j1]*np.power(-1,i)) + (lat[i1,j1]*np.power(-1,i1)) 
            tempy = (lat[i,j]*np.power(-1,j)) + (lat[i1,j]*np.power(-1,j)) 
            + (lat[i,j1]*np.power(-1,j1)) + (lat[i1,j1]*np.power(-1,j1)) 
            
            if(np.abs(tempx)>np.abs(tempy)):
                if(tempx>0):
                    latz[i][j] = 0
                if(tempx<0):
                    latz[i][j] = np.pi
            
            if(np.abs(tempx)<np.abs(tempy)):
                if(tempy>0):
                    latz[i][j] = np.pi*0.5
                if(tempy<0):
                    latz[i][j] = 1.5*np.pi
            
            if(np.abs(tempx)==np.abs(tempy)):
                ran = np.random.uniform(0,1)
                if(ran<0.5):
                    #x
                    if(tempx>0):
                        latz[i][j] = 0
                    if(tempx<0):
                        latz[i][j] = np.pi
                if(ran>=0.5):#greater than equal to?
                    #y
                    if(tempy>0):
                        latz[i][j] = np.pi*0.5
                    if(tempy<0):
                        latz[i][j] = 1.5*np.pi
        
    return latz 

@jit(nopython=True)
def vtx(latz):#gets the defects in the z4 model
    s = int(len(latz))
    defs = np.zeros((s,s))
    nv = 0
    nav = 0
    for i in range(0,s):
        for j in range(0,s):
            theta12 = latz[i][(j-1)%s] - latz[i][j]
            theta23 = latz[(i+1)%s][(j-1)%s] - latz[i][(j-1)%s]
            theta34 = latz[(i+1)%s][j] - latz[(i+1)%s][(j-1)%s]
            theta41 = latz[i][j] - latz[(i+1)%s][j]#this was i before
            
            if(theta12>np.pi):
                theta12 = theta12 - 2*np.pi
            if(theta12<-1*np.pi):
                theta12 = theta12 + 2*np.pi
            
            if(theta23>np.pi):
                theta23 = theta23 - 2*np.pi
            if(theta23<-1*np.pi):
                theta23 = theta23 + 2*np.pi
                
            if(theta34>np.pi):
                theta34 = theta34 - 2*np.pi
            if(theta34<-1*np.pi):
                theta34 = theta34 + 2*np.pi
                
            if(theta41>np.pi):
                theta41 = theta41 - 2*np.pi
            if(theta41<-1*np.pi):
                theta41 = theta41 + 2*np.pi
                
            theta = theta12 + theta23 + theta34 + theta41
            
            if(theta==2*np.pi):
                nv = nv + 1
                defs[i][j] = 1
            
            if(theta==-2*np.pi):
                nav = nav + 1
                defs[i][j] = -1
    
    return nv, nav, defs

@jit(nopython=True)
def domg(lat): #looks only for the clean phases and adds the boundary terms using a majority rule
    s = int(len(lat))
    latz = np.zeros((s,s))
    for i in range(0,s):
        for j in range(0,s):
            i1 = (i+1)%s
            j1 = (j+1)%s
            tempx = float ((lat[i,j]*np.power(-1,i)) + (lat[i1,j]*np.power(-1,i1))  + (lat[i,j1]*np.power(-1,i)) + (lat[i1,j1]*np.power(-1,i1)))
            tempy = float ((lat[i,j]*np.power(-1,j)) + (lat[i1,j]*np.power(-1,j)) + (lat[i,j1]*np.power(-1,j1)) + (lat[i1,j1]*np.power(-1,j1)))
            if(tempx==-4):
                latz[i,j] = -1
            if(tempx==4):
                latz[i,j] = 1
            if(tempy==-4):
                latz[i,j] = -2
            if(tempy==4):
                latz[i,j] = 2
    for i in range(0,s):
        for j in range(0,s):
            flag = 0
            if(latz[i,j]!=0):
                i1 = (i+1)%s
                j1 = (j+1)%s
                i2 = (i-1)%s
                j2 = (j-1)%s
                if((latz[i,j]==latz[i1,j]) or (latz[i,j]==latz[i2,j]) or (latz[i,j]==latz[i,j1])
                or (latz[i,j]==latz[i,j2])):
                    flag = -1
                if(flag==0):
                    latz[i,j] = 0

    for i in range(0,s):
        for j in range(0,s):
            if(latz[i,j]==0):
                flag = 0
                for arr in nij:
                    di = arr[0]
                    dj = arr[1]
                    if(latz[(i+di)%s,(j+dj)%s]!=0):
                        flag = -1
                        break
                    else:
                        continue
                    
                if(flag==-1):
                    i1 = (i+1)%s
                    j1 = (j+1)%s
                    tempx = (lat[i,j]*np.power(-1,i)) + (lat[i1,j]*np.power(-1,i1))  + (lat[i,j1]*np.power(-1,i)) + (lat[i1,j1]*np.power(-1,i1)) 
                    tempy = (lat[i,j]*np.power(-1,j)) + (lat[i1,j]*np.power(-1,j)) + (lat[i,j1]*np.power(-1,j1)) + (lat[i1,j1]*np.power(-1,j1)) 
                    if(np.abs(tempx)>np.abs(tempy)):
                        if(tempx>0):
                            latz[i,j] = 1
                        if(tempx<0):
                            latz[i,j] = -1
            
                    if(np.abs(tempx)<np.abs(tempy)):
                        if(tempy>0):
                            latz[i,j] = 2
                        if(tempy<0):
                            latz[i,j] = -2
                    
                    if(np.abs(tempx)==np.abs(tempy)):
                        ran = np.random.uniform(0,1)
                        if(ran<0.5):
                            #x
                            if(tempx>0):
                                latz[i,j] = 1
                            if(tempx<0):
                                latz[i,j] = -1
                        if(ran>=0.5):#greater than equal to?
                            #y
                            if(tempy>0):
                                latz[i,j] = 2
                            if(tempy<0):
                                latz[i,j] = -2
                else:
                    continue
                        
    return latz

@jit(nopython=True)
def defcount(latz):
    s = int(len(latz))
    defs = np.zeros((s,s))
    count = 0
    for i in range(0,s):
        for j in range(0,s):
            a = 0
            b = 0
            c = 0
            d = 0
            for di in range(0,2):
                for dj in range(0,2):
                    if(latz[(i+di)%s][(j+dj)%s]==1):
                        a = a+1
                    if(latz[(i+di)%s][(j+dj)%s]==-1):
                        b = b+1
                    if(latz[(i+di)%s][(j+dj)%s]==2):
                        c = c+1
                    if(latz[(i+di)%s][(j+dj)%s]==-2):
                        d = d+1
            if((a*b*c*d)!=0):
                defs[i][j] = 1
                count = count + 1
    
    return defs,count


