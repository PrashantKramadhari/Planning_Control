#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from misc import Timer
from world import BoxWorld
from scipy.spatial import distance
import math


# # Define the planning world

# Define world with obstacles.



world = BoxWorld([[0, 10], [0, 10]])
world.add_box(2, 2, 6, 6)
world.add_box(1, 6, 4, 3)
world.add_box(4, 1, 5, 4)

plt.figure(10, clear=True)
world.draw()
plt.xlabel('x')
plt.ylabel('y')
plt.axis([world.xmin, world.xmax, world.ymin, world.ymax]);


# # Car simulation function

# Define function needed to simulate motin of single-track model.

def sim_car(xk, u, step, h = 0.01, L=2, v=15):
    """Car simulator
    
    Simulate car motion forward in time from state xk with time-step length step. Returns next state.
    
    x' = v*cos(th)
    y' = v*sin(th)
    th' = v*tan(phi)/L
    phi = u
    x = [x y th]
    """

    # Simulation with discretization using forward Euler

    t = 0
    N = np.int(step / h) + 1
    states = np.zeros((3, N))
    states[:, 0] = xk

    k = 0
    while k < N-1:
        hk = min(h, step - t)
        states[0, k+1] = states[0, k] + hk * v * math.cos(states[2, k])
        states[1, k+1] = states[1, k] + hk * v * math.sin(states[2, k])
        states[2, k+1] = states[2, k] + hk * v * math.tan(u) / L
        t = t + h
        k = k + 1

    return states   


# # Implementation of RRT for kinematic car model
# 
# Car model has two translational degrees of freedom and one orientational.

def rrt_diff(start, goal, u_c, sim, w, opts):
    """RRT planner for kinematic car model
    
    Input arguments:
        start - initial state
        goal - desired goal state
        u_c - vector with possible control actions (steering angles)
        sim - function reference to the simulation model of the car motion
        world - description of the map of the world
                using an object from the class BoxWorld
        opts - structure with options for the RRT
    
    Output arguments:
        goal_idx - index of the node closest to the desired goal state
        nodes - 2 x N matrix with each column representing a state j
                in the tree
        parents - 1 x N vector with the node number for the parent of node j 
                  at element j in the vector (node number counted as column
                  in the matrix nodes)
        trajectories - a struct with the trajectory segment for reaching
                       node j at element j (node number counted as column
                       in the matrix nodes)
        Tplan - the time taken for computing the plan    
    """

    rg = np.random.default_rng()
    def SampleFree():
        """Returns a sample state x in the free space"""
        if rg.uniform(0, 1) < opts['beta']:
            return np.array(goal)
        else:
            foundRandom = False
            th = rg.uniform(0, 1) * 2 * np.pi - np.pi
            while not foundRandom:
                p = (rg.uniform(0, 1, 2) * [w.xmax - w.xmin, w.ymax-w.ymin] + 
                     [w.xmin, w.ymin])
                if w.obstacle_free(p[:, None]):
                    foundRandom = True
        return np.array([p[0], p[1], th])

    def Nearest(x):
        """Returns index of state nearest to x in nodes"""

        idx = np.argmin(
            distance.cdist(
                nodes.T, x[:, None].T, 'sqeuclidean'))
        return idx    
    
    def SteerCandidates(x_nearest, x_rand):
        """Compute all possible paths for different steering control signals u_c to move from x_nearest towards x_rand"""
        
        new_paths = [sim(x_nearest, ui, opts['delta']) for ui in u_c]
        new_free = np.where([w.obstacle_free(traj_i) for traj_i in new_paths])[0]
        valid_new_paths = [new_paths[i] for i in new_free]
        
        if np.any(new_free):
            dist_to_x_rand = [np.linalg.norm(xi[:, -1] - x_rand) for xi in valid_new_paths]
        else:
            dist_to_x_rand = -1

        return valid_new_paths, dist_to_x_rand

    
    def DistanceFcn(x1, x2):
        return np.linalg.norm(x1 - x2)


    # Start time measurement and define variables for nodes, parents, and 
    # associated trajectories
    T = Timer()
    T.tic()
    nodes = np.array(start).reshape((-1, 1))
    parents = [0]
    state_trajectories = [start]

    # YOUR CODE HERE
    ###########################
    for i in range(opts['K']):
        q_rand = SampleFree()
        q_nearest = Nearest(q_rand)
        q_new = SteerCandidates(nodes[:,q_nearest],q_rand)
        if q_new[1] != -1:
            idx = np.argmin(q_new[1]) #find the smallest distance
            q_new_best = q_new[0][idx]
            if w.ObstacleFree_2_pt(nodes[:,q_nearest][:, None], q_new_best[:,-1][:, None]) :
                nodes = np.hstack((nodes,q_new_best[:,-1][:, None]))
                parents.append(q_nearest)
                state_trajectories.append(q_new_best)
    ###########################
    
    Tplan = T.toc()
    goal_idx = np.argmin(np.sum((nodes - np.array(goal)[:, None])**2, axis=0))    
    return goal_idx, nodes, parents, state_trajectories, Tplan


start = [1, 1, np.pi / 4]
goal = [9, 9, np.pi / 2]

u_c = np.linspace(-np.pi / 4, np.pi / 4, 10)

opts = {'beta': 0.05,  # Probability for selecting goal state as target state
        'eps': -0.01,  # Threshold for stopping the search
        'delta': 0.1,  # Step size (in time)
        'K': 4000}

goal_idx, nodes, parents, state_trajectories, T = rrt_diff(start, goal, u_c, sim_car, world, opts)

plt.plot(start[0], start[1], '+k') #Intial goal in back
plt.plot(goal[0], goal[1], '+k') #target goal in black


drawlines = []
idx = goal_idx
while idx != 0:
    ll = np.column_stack((nodes[:, parents[idx]], nodes[:, idx]))
    drawlines.append(ll[0])
    drawlines.append(ll[1])
    idx = parents[idx]
    plt.plot(*drawlines,color='b', lw=2)
plt.show()
print(f'Finished in {T:.2f} sek')


