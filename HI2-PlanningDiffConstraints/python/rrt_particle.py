#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from misc import Timer
from world import BoxWorld
from queues import LIFO,FIFO,PriorityQueue


# # Define the planning world

# Define the world with some obstacles
world = BoxWorld([[0, 10], [0, 10]])
world.add_box(2, 2, 6, 6)
world.add_box(1, 6, 4, 3)
world.add_box(4, 1, 5, 4)

plt.figure(10, clear=True)
world.draw()
plt.xlabel('x')
plt.ylabel('y')
plt.axis([world.xmin, world.xmax, world.ymin, world.ymax]);


# # Implementation of RRT

# Implementation of the RRT planning algorithm

def rrt_particle(start, goal, w, opts):
    """RRT planner for particle moving in a 2D world
    
    Input arguments:
        start - initial state
        goal - desired goal state
        world - description of the map of the world
                using an object from the class BoxWorld
        opts - structure with options for the RRT*

    Output arguments:
        goal_idx - index of the node closest to the desired goal state
        nodes - 2 x N matrix with each column representing a state j
                in the tree
        parents - 1 x N vector with the node number for the parent of node j 
                  at element j in the vector (node number counted as column
                  in the matrix nodes)
        Tplan - the time taken for computing the plan        
    """
    rg = np.random.default_rng()  # Get the random number generator
    def SampleFree():
        """Sample a state x in the free state space"""

        if rg.uniform(0, 1, 1) < opts['beta']:
            return np.array(goal)
        else:
            foundRandom = False
            while not foundRandom:
                x = (rg.uniform(0, 1, 2) * [w.xmax - w.xmin, w.ymax - w.ymin] + 
                     [w.xmin, w.ymin])
                if w.obstacle_free(x[:, None]):
                    foundRandom = True
            return x

    def Nearest(x):
        """Find index of state nearest to x in the matrix nodes"""        
        idx = np.argmin(np.sum((nodes - x[:, None])**2, axis=0))
        return idx
    
    def Steer(x1, x2):
        """Steer from x1 towards x2 with step size opts['delta']
        
        If the distance to x2 is less than opts['delta'], return
        state x2.
        """        
        
        dx = np.linalg.norm(x2 - x1)
        if dx < opts['delta']:
            x_new = x2
        else:
            x_new = x1 + opts['delta'] * (x2 - x1) / dx
        return x_new

    
    # Start time measurement and define variables for nodes and parents
    T = Timer()
    T.tic()
    nodes = np.array(start).reshape((-1, 1))  # Make numpy column vector
    parents = [0]

    # YOUR CODE HERE
    ###########################
    for i in range(opts['K']):
        q_rand = SampleFree()
        q_nearest = Nearest(q_rand)
        q_new = Steer(nodes[:,q_nearest],q_rand)
        if w.ObstacleFree_2_pt(nodes[:,q_nearest][:, None], q_new[:, None]) :
            nodes = np.hstack((nodes,q_new.reshape(-1, 1)))
            parents.append(q_nearest)
    ###########################
    Tplan = T.toc()
    idx_goal = np.argmin(np.sum((nodes - np.array(goal).reshape((-1, 1)))**2, axis=0))
    
    return idx_goal, nodes, parents, Tplan


# Run the planner

start = np.array([1, 1])
goal = np.array([9, 9])

opts = {'beta': 0.05,  # Probability for selecting goal state as target state
        'delta': 0.1,  # Step size
        'eps': -0.01,  # Threshold for stopping the search
        'K': 5000}     # Maximum number of iterations

print('Planning ...')
idx_goal, nodes, parents, T = rrt_particle(start, goal, world, opts)
print('Finished in {:.2f} sek'.format(T))

if (nodes[:,idx_goal][0]== goal[0]) and nodes[:,idx_goal][1]== goal[1]:
    print("found goal")

plt.plot(nodes[:,:][0], nodes[:,:][1], '.g')
plt.plot(start[0], start[1], 'ok') #Intial goal in back
plt.plot(goal[0], goal[1], 'ok') #target goal in black
plt.plot(nodes[:,idx_goal][0], nodes[:,idx_goal][1], 'xb') #reached goal in blue

""" Hint on plotting: To plot the path corresponding to the found solution,
the following code could be useful (utilizing backtracking from the goal
node: """

drawlines = []
idx = idx_goal
while idx != 0:
    ll = np.column_stack((nodes[:, parents[idx]], nodes[:, idx]))
    drawlines.append(ll[0])
    drawlines.append(ll[1])
    idx = parents[idx]
    plt.plot(*drawlines,color='b', lw=2)
#_, ax = plt.subplots(num=99, clear=True)
#ax.plot(*drawlines, color='b', lw=2)

plt.show()

print()
# # Plots and analysis



