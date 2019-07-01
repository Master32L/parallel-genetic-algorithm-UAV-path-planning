import numpy as np

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

import math



def parallel_GA(cluster, pop_size=1024, tpb=32, seed=1, num_iter=100,
                num_elites=32, tournament_size=4, mutate_prob=0.08):
    r"""Solve TSP problem formed by Control Points (CPs) in the cluster using
    parallel genetic algorithm.

    Args:
        cluster(pandas.DataFrame(m, 2)): Data of CPs in the cluster.
        pop_size(int): Population size of each generation. Default: 1024
        tpb(int): Thread per block for kernels. Default: 32
        seed(int): Random seed for the rng states generator. Default: 1
        num_iter(int): Number of GA iterations. Default: 100
        num_elites(int): Number of elites in each iteration. Default: 32
        tournament_size(int): Size of tournament. Default: 4.
        mutate_prob(float): Probability of each idv to mutate. Default: 0.08
    """
    # compute problem parameters
    m = len(cluster) # number of CPs
    bpg = (pop_size-1)//tpb + 1 # block per grid for kernels

    # compute random states for each thread
    rng_states = create_xoroshiro128p_states(tpb*bpg, seed=seed)

    # form data into numpy array and send to device
    coordinates = cluster.to_numpy()
    assert(coordinates.shape == (m,2))
    assert(coordinates.dtype == np.float32)
    d_coordinates = cuda.to_device(coordinates)

    # generate initial population and send to device
    d_pop_init = cuda.device_array((pop_size,m), dtype=np.float32)
    initialize_kernel[tpb, bpg](d_pop_init, rng_states)
    pop = d_pop_init.copy_to_host()
    pop = np.argsort(pop, axis=1).astype(np.int32)
    d_pop = cuda.to_device(pop)

    # evolve
    d_fitness_all = cuda.device_array(pop_size, dtype=np.float32)
    all_pop = [pop]
    all_fitness = []
    for i in range(num_iter):
        # evaluate fitness
        fitness_kernel[tpb, bpg](d_pop, d_coordinates, d_fitness_all)
        # sort
        fitness_all = d_fitness_all.copy_to_host()
        ranking = np.argsort(fitness_all).astype(np.int32) # indices of idv with fitness value from low to high
        is_elite = [(idv in ranking[:num_elites]) for idv in np.arange(pop_size)] # boolean
        d_is_elite = cuda.to_device(np.array(is_elite, dtype=np.bool))
        d_nonelite = cuda.to_device(ranking[num_elites:])
        all_fitness.append(fitness_all)
        # evolve
        d_next_gen = cuda.device_array_like(pop)
        cross_over_kernel[tpb,bpg](d_pop, d_is_elite, d_nonelite, d_next_gen,
                                   num_elites, rng_states, tournament_size,
                                   d_fitness_all)
        mutate_kernel[tpb,bpg](d_next_gen, d_is_elite, rng_states, mutate_prob)
        if i < 32:
            local_opt_kernel[tpb,bpg](d_next_gen, d_is_elite, d_coordinates)
        d_pop = d_next_gen
        all_pop.append(d_pop.copy_to_host())

    # result
    fitness_kernel[tpb, bpg](d_pop, d_coordinates, d_fitness_all)
    fitness_all = d_fitness_all.copy_to_host()
    all_fitness.append(fitness_all)
    return all_pop, all_fitness



@cuda.jit
def initialize_kernel(d_pop_init, rng_states):
    r"""Generate random numbers.
    """
    i = cuda.grid(1)
    pop_size = d_pop_init.shape[0]
    m = d_pop_init.shape[1]
    if i < pop_size:
        for j in range(m):
            rnd = xoroshiro128p_uniform_float32(rng_states, i)
            d_pop_init[i,j] = rnd



@cuda.jit
def fitness_kernel(d_pop, d_coordinates, d_fitness_all):
    r"""Evaluate total distance of each individual.
    """
    i = cuda.grid(1)
    pop_size = d_pop.shape[0]
    m = d_pop.shape[1]
    if i < pop_size:
        fitness = 0
        for j in range(m-1):
            fitness += distance(d_pop[i,j], d_pop[i,j+1], d_coordinates)
        d_fitness_all[i] = fitness + distance(d_pop[i,0], d_pop[i,m-1],
                                              d_coordinates)



@cuda.jit(device=True)
def distance(cp1, cp2, d_coordinates):
    r"""Compute Euclidean distance between two CPs.
    """
    lat1 = d_coordinates[cp1,0]
    lng1 = d_coordinates[cp1,1]
    lat2 = d_coordinates[cp2,0]
    lng2 = d_coordinates[cp2,1]
    return math.sqrt((lat1-lat2)**2+(lng1-lng2)**2)



@cuda.jit
def cross_over_kernel(d_pop, d_is_elite, d_nonelite, d_next_gen, num_elites,
                      rng_states, tournament_size, d_fitness_all):
    r"""Pass on elites and perform crossover for others.
    """
    i = cuda.grid(1)
    pop_size = d_pop.shape[0]
    if i < pop_size:
        if d_is_elite[i]: # copy
            for j in range(d_pop.shape[1]):
                d_next_gen[i,j] = d_pop[i,j]
        else: # crossover
            parent1 = tournament(rng_states, i, pop_size, d_nonelite,
                                 num_elites, d_fitness_all, tournament_size)
            parent2 = tournament(rng_states, i, pop_size, d_nonelite,
                                 num_elites, d_fitness_all, tournament_size)
            cross_over_1p(parent1, parent2, rng_states, d_pop, d_next_gen, i)



@cuda.jit(device=True)
def tournament(rng_states, i, pop_size, d_nonelite, num_elites,
               d_fitness_all, tournament_size):
    r"""Randomly choose candidates from nonelite individuals then choose the
    best as one parent.
    """
    rnd = xoroshiro128p_uniform_float32(rng_states, i)
    num_nonelite = d_nonelite.size
    parent = d_nonelite[int(math.floor(rnd*num_nonelite))]
    min_fitness = d_fitness_all[parent]
    for j in range(tournament_size-1):
        rnd = xoroshiro128p_uniform_float32(rng_states, i)
        new_parent = parent = d_nonelite[int(math.floor(rnd*num_nonelite))]
        if min_fitness < d_fitness_all[new_parent]:
            parent = new_parent
            min_fitness = d_fitness_all[new_parent]
    return parent



@cuda.jit(device=True)
def cross_over_1p(parent1, parent2, rng_states, d_pop, d_next_gen, i):
    r"""Perform 1-point crossover. Copy a portion of parent1, then fill the
    rest with parent2.
    """
    m = d_pop.shape[1]
    rnd = xoroshiro128p_uniform_float32(rng_states, i)
    split = int(math.floor(rnd*m))
    # copy from parent1
    for j in range(split):
        d_next_gen[i,j] = d_pop[parent1,j]
    # copy from parent2
    idx = split
    for j in range(m):
        cp2 = d_pop[parent2,j]
        repeat = False
        for k in range(split):
            cp1 = d_next_gen[i,k]
            if cp1 == cp2:
                repeat = True
                break
        if repeat == False:
            d_next_gen[i,idx] = cp2
            idx += 1
            if idx == m:
                break



@cuda.jit
def mutate_kernel(d_next_gen, d_is_elite, rng_states, mutate_prob):
    r"""Perform mutation by randomly swapping two CPs.
    """
    i = cuda.grid(1)
    if i < d_next_gen.shape[0]:
        if d_is_elite[i] == False:
            rnd = xoroshiro128p_uniform_float32(rng_states, i)
            if rnd < mutate_prob:
                rnd = xoroshiro128p_uniform_float32(rng_states, i)
                idx1 = int(math.floor(rnd*d_next_gen.shape[1]))
                rnd = xoroshiro128p_uniform_float32(rng_states, i)
                idx2 = int(math.floor(rnd*d_next_gen.shape[1]))
                tmp = d_next_gen[i,idx1]
                d_next_gen[i,idx1] = d_next_gen[i,idx2]
                d_next_gen[i,idx2] = tmp



@cuda.jit
def local_opt_kernel(d_next_gen, d_is_elite, d_coordinates):
    r"""Perform 2-opt for each individual.
    """
    i = cuda.grid(1)
    if i < d_next_gen.shape[0]:
        if d_is_elite[i] == False:
            improved = True
            count = 0
            while count < d_next_gen.shape[0] and improved:
                improved = local_opt_one_trial(d_next_gen, d_coordinates, i)
                count += 1



@cuda.jit(device=True)
def local_opt_one_trial(d_next_gen, d_coordinates, i):
    r"""Perform one trial of 2-opt.
    """
    m = d_next_gen.shape[1]
    for x in range(m-1):
        a = d_next_gen[i,x]
        b = d_next_gen[i,x+1]
        for y in range(x+2, m-1):
            c = d_next_gen[i,y]
            d = d_next_gen[i,y+1]
            if (distance(a,b,d_coordinates)+distance(c,d,d_coordinates) >
                distance(a,c,d_coordinates)+distance(b,d,d_coordinates)):
                p = x + 1
                q = y
                while q - p >= 1:
                    tmp = d_next_gen[i,p]
                    d_next_gen[i,p] = d_next_gen[i,q]
                    d_next_gen[i,q] = tmp
                    p += 1
                    q -= 1
                return True
    return False
