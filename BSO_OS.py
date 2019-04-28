import numpy as np
import math
import matplotlib.pyplot as plt

def benchmark_func(vec,index):
    dim = len(vec)
    if index == 1: # Rosenbrock function
        f = 100 * np.sum((vec[0:dim-1]**2 - vec[1:dim])**2) + np.sum((vec[0:dim-1] - 1)**2)
    if index == 2: # Ackley function
        f = -20 * np.exp(-0.2 * np.sqrt(np.sum(vec**2)/dim)) - np.exp(np.sum(np.cos(2*math.pi*vec))/dim) + 20 + np.exp(1)
    if index == 3: # Schwefel's P226
        f = -np.sum(vec * np.sin(np.sqrt(np.abs(vec)))) + 418.9829 * dim
    
    return f

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
# Problem parameters
index = 2
dim = 5
ubound = 100
lbound = -100
# BSO-OS hyper-parameters
maxe = dim * 10000 # max number of evaluation
num = 100 # number of sampling
pe = 0.2 # percentage_elitist
pdd = 0.2 # probability to determine whether a dimension is disrupted or not
pd = 1 # probability for disrupting elitists. one elitis every 5 generations, and only one dimension
pre = 0.2 # probability for select elitist, not normals, to generate new individual
po = 0.8 # probability for select one individual, not two, to generate new individual
ls = maxe / 20 # slope of the s-shape function
Lambda = 0.5 # decay factor
nelite = np.round(num * pe) # number of elitist
nnorm = num - nelite # number of normals
# Initialization
step_size = np.ones(dim) # step size for search
archi = lbound + (ubound - lbound) * np.random.rand(num,dim) # archieve to store sampling solutions
tem_archi = np.ones((num,dim)) # temporary archieve

ne = 0 # counter for number of evalution
bestf = [] # store best fitness for each iteration
fitness_sol = np.ones(num) # store
tem_sol = np.ones(dim)
# caluculate fitness for each solution in the initialized archieve
for i in range(0,num):
    fitness_sol[i] = benchmark_func(archi[i,:],index)

# start the main loop of the BSO algorithm
cycle = 0
while(ne < maxe):
    cycle += 1
    if cycle > 1: # do not do disruption for the first iteration
        # disrupt every iteration but for one dim of one solution
        r_1 = np.random.rand()
        if r_1 < pd: # decide whether to select one individual to be disrupted
            idx = int(np.floor(num * np.random.rand()))
            tem_sol = archi[idx,:]
            tem_sol[int(np.floor(dim * np.random.rand()))] = lbound + (ubound - lbound) * np.random.rand()
            fv = benchmark_func(tem_sol,index)
            ne += 1
            archi[idx,:] = tem_sol
            fitness_sol[idx] = fv
    # sort solutions in an archieve based on their function value
    idxsort = np.argsort(fitness_sol)
    # record the best function value in each generation
    bestf.append(fitness_sol[idxsort[0]])
    # calculate s-shape function
    mu = sigmoid((Lambda * maxe - ne)/ls)
    
    # generate num new solutions by adding Gaussian random values
    for i in range(0,num):
        r_1 = np.random.rand()
        if r_1 < pre: # select elitists to generate a new solution
            r = np.random.rand()
            ind_one = int(np.floor(nelite * np.random.rand()))
            ind_two = int(np.floor(nelite * np.random.rand()))
            while(ind_one == ind_two):
                ind_one = int(np.floor(nelite * np.random.rand()))
                ind_two = int(np.floor(nelite * np.random.rand()))
            if r < po: # use one elitist
                tem_sol = archi[idxsort[int(ind_one)]]
            else: # use two elitists
                rat = np.random.rand()
                tem_sol = rat * archi[idxsort[ind_one]] + (1-rat) * archi[idxsort[ind_two]]
        else: # select normals to generate a new solution
            r = np.random.rand()
            ind_one = int(num-1-np.floor(nelite * np.random.rand()))
            ind_two = int(num-1-np.floor(nelite * np.random.rand()))
            while(ind_one == ind_two):
                ind_one = int(num-1-np.floor(nelite * np.random.rand()))
                ind_two = int(num-1-np.floor(nelite * np.random.rand()))
            if r < po:
                tem_sol = archi[idxsort[ind_one]]
            else:
                rat = np.random.rand()
                tem_sol = rat * archi[idxsort[ind_one]] + (1-rat) * archi[idxsort[ind_two]]
        
        # add Gaussian disturbance to the tem_sol to generate a new solution
        step_size = mu * np.random.rand(dim)
        tem_sol += step_size * np.random.normal(0, 1, dim)
        # selection between new one and the old one with the same index in archieve
        fv = benchmark_func(tem_sol,index)
        ne += 1
        if fv < fitness_sol[i]: # reserve the better one
            fitness_sol[i] = fv
            archi[i,:] = tem_sol
    print(ne)
    
plt.figure()
plt.plot(bestf)
plt.xlabel('number of iteration')
plt.ylabel('log(loss function)')
plt.yscale('log')
plt.grid()
plt.savefig('exp',dpi=600)
plt.show()