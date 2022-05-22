import numpy as np
import math

global j1
global T
global s
global j2
global dis_prob
global g
global Tc
global rij
global lind
global nij
global mcsteps

j1 = 1
dis_prob = 0
g = 1
j2 = -1*g*j1
s = 32
Tc = 1.11*j1

nij = np.array([0,0])
for i in range(-1,2):
    for j in range(-1,2):
        nij = np.vstack([nij,np.array([i,j])])

nij = nij[1:,:]


rij = np.array([])
pij = np.array([])


lind = np.array([0,0])
for i in range(0,s):
    for j in range(0,s):
        lind = np.vstack([lind,np.array([i,j])])

lind = lind[1:,:]

def func(x, a, b, c):
    return a*np.exp(-b * x) + c
 
def func1(x,a,b,c):
    #print((a*np.float_power(x,np.ones(len(x),dtype=np.float32)*(-b))) + c)
    return (a*np.float_power(x,np.ones(len(x),dtype=np.float32)*(-b))) + c
    #return (np.float_power(x,np.ones(len(x),dtype=np.float32)*(-b))) 

def func2(x, m, c):
    return m*x + c
