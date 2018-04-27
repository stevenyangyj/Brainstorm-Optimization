# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 16:28:00 2018

BSO-OS
"""
import numpy as np

def benchmark_func(vec):
    dim = len(vec)
    f = 100 * np.sum((vec[0:dim-1]**2 - vec[1:dim])**2) + np.sum((vec[0:dim-1] - 1)**2)
    
    return f
# Problem parameters
dim = 30
ubound = 30
lbound = -30
# BSO-OS hyper-parameters
maxe = 10000 * dim
num = 100
pe = 0.2 # percentage_elitist
pdd = 0.2 # probability to determine whether a dimension is disrupted or not
pd = 1 # probability for disrupting elitists. one elitis every 5 generations, and only one dimension
pre = 0.2 # probability for select elitist, not normals, to generate new individual
po = 0.8 # probability for select one individual, not two, to generate new individual
ls = maxe / 20 # slope of the s-shape function
Lambda = 0.5 # decay factor
nelite = np.floor(num * pe) # number of elitist
nnorm = num - nelite # number of normals
# Initialization
step_size = np.ones(dim) # step size for search
archi = lbound + (ubound - lbound) * np.random.rand(num,dim) # archieve yo store sampling solutions
tem_archi = np.ones((num,dim)) # temporary archieve

ne = 0 # counter for number of evalution
bestf = [] # store best fitness for each iteration
fitness_sol = np.ones(num) # store
tem_sol = np.ones(dim)
# caluculate fitness for each solution in the initialized archieve
for i in range(0,num):
    fitness_sol[i] = benchmark_func(archi[i,:])

# start the main loop of the BSO algorithm
cycle = 0
while(ne < maxe):
    cycle += 1
    if cycle > 1: # do not do disruption for the first iteration
        # disrupt every iteration but for one dim of one solution