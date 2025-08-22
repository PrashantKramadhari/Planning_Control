#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from misc import Timer
from world import BoxWorld


# # Define the planning world

# Define world with obstacles.

w = BoxWorld([[0, 10], [0, 10]])
w.add_box(2, 2, 6, 6)
w.add_box(1, 6, 4, 3)
w.add_box(4, 1, 5, 4)

plt.figure(10, clear=True)
w.draw()
plt.axis([w.xmin, w.xmax, w.ymin, w.ymax]);


# # Implementation of RRT* planning algorithm

def rrt_star_particle(start, goal, w, opts):
    def SampleFree():
        """Sample a state x in the free state space"""
        if np.random.uniform(0, 1, 1) < opts['beta']:
            return goal
        else:
            foundRandom = False
            while not foundRandom:
                x = np.random.uniform(0, 1, 2)*[w.xmax-w.xmin, w.ymax-w.ymin] + [w.xmin, w.ymin]
                if w.obstacle_free(x[:, None]):
                    foundRandom = True
        return x

    def Nearest(x):
        """Return index of state nearest to x in nodes"""
        idx = np.argmin(np.sum((nodes-x[:, None])**2, axis=0))
        return idx
    
    def Near(x, r):
        """Return the indices of the states in nodes within a neighborhood with radius r from state x"""
        idx = np.where(np.sum((nodes-x[:, None])**2, axis=0) < r**2)
        return idx[0]

    def Steer(x1, x2):
        """Steering function for moving from x1 to x2"""
        dx = np.linalg.norm(x2 - x1)
        if dx < opts['delta']:
            x_new = x2
        else:
            x_new = x1 + opts['delta']*(x2-x1)/dx
        return x_new
    
    def ConnectMinCost(x_new, near_idx, idx_nearest, cost_nearest):
        """Connecting along a path from x_nearest to x_new with
           minimum cost among the states in a neighborhood of x_nearest
           described by the indices near_idx in nodes"""
        
        idx_min = idx_nearest
        cost_min = cost_nearest

        for idx_n in near_idx:
            x_near = nodes[:, idx_n]

            if (x_new[0] == x_near[0]) and (x_new[1] == x_near[1]):
                p = x_new[:, None]
            else:
                p = np.row_stack((np.arange(x_near[0], x_new[0], (x_new[0] - x_near[0])/10),
                                  np.arange(x_near[1], x_new[1], (x_new[1] - x_near[1])/10)))
            cost_near = cost[idx_n] + np.linalg.norm(x_near - x_new)

            if cost_near < cost_min and w.ObstacleFree(p):
                cost_min = cost_near
                idx_min = idx_n
        return idx_min, cost_min

    def RewireNeighborhood(x_new, near_idx, cost_min):
        """Function for (possible) rewiring of the nodes in the neighborhood
           described by the indices near_idx on nodes via the new state x_new,
           if a path with less cost could be found"""
        for idx_n in near_idx:
            x_near = nodes[:, idx_n]
            
            if (x_new[0] == x_near[0]) and (x_new[1] == x_near[1]):
                p = x_new[:, None]
            else:
                p = np.row_stack((np.arange(x_near[0], x_new[0], (x_new[0] - x_near[0])/10),
                                  np.arange(x_near[1], x_new[1], (x_new[1] - x_near[1])/10)))
            cost_near = cost_min + np.linalg.norm(x_near - x_new)
            if cost_near < cost[idx_n] and w.ObstacleFree(p):
                parents[idx_n] = len(parents)-1
                cost[idx_n] = cost_near
    
    # Start time measurement and define variables for nodes, parents, and 
    # associated cost
    T = Timer()
    T.tic()
    nodes = start.reshape((-1, 1))
    parents = np.array([0], dtype=np.int)
    cost = np.array([0])
    
    # YOUR CODE HERE
    ###########################
    for i in range(opts['K']):
        q_rand = SampleFree()
        q_nearest = Nearest(q_rand)
        q_new = Steer(nodes[:,q_nearest],q_rand)
        if w.ObstacleFree_2_pt(nodes[:,q_nearest][:, None], q_new[:, None]) :
            Q_near = Near(q_new ,opts["r_neighbor"])                                                                    
            nodes = np.hstack((nodes,q_new.reshape(-1, 1)))
            q_min_idx, c_min = ConnectMinCost(q_new, Q_near, q_nearest, cost[q_nearest])
            cost = np.hstack((cost, c_min))
            parents = np.hstack((parents, q_min_idx))
            RewireNeighborhood(q_new, Q_near, c_min)
    ###########################
    Tplan = T.toc()
    idx_goal = np.argmin(np.sum((nodes - np.array(goal).reshape((-1, 1)))**2, axis=0))

    return idx_goal, nodes, parents, Tplan


start = np.array([1, 1])
goal = np.array([9, 9])

opts = {'beta': 0.01,  # Probability for selecting goal state as target state
        'delta': 0.3,  # Step size
        'eps': -0.01,  # Threshold for stopping the search
        'r_neighbor': 0.2,
        'K': 5000}

idx_goal, nodes, parents, T = rrt_star_particle(start, goal, w, opts)
print(f'Finished in {T:.3f} sek')

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