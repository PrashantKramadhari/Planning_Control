#!/usr/bin/env python
# coding: utf-8

# # TSFS12 Hand-in exercise 1: Discrete planning in structured road networks
# Erik Frisk (erik.frisk@liu.se)

# Do initial imports of packages needed


import numpy as np
import matplotlib.pyplot as plt
from misc import Timer, LatLongDistance
from queues import FIFO, LIFO, PriorityQueue
from osm import loadOSMmap


import pickle
tmp = pickle.load( open( "linkoping.osm.pickle", "rb" ) )

""" ######TEST##################

q = PriorityQueue()
q. insert (3, 10)
q. insert (2, 20)
q. insert (1, 30)
q. insert (1, 20)

q.size ()
print(q.peek())
prio, value = q.pop()
print(f'Prior = {prio}, Val = {value}')
print("Test finished")
############################ """


flag_planner = 15  #DFS :1 BFS:2  Dijkstra:3  Astar4: BestFirst:5 All:15

dataDir = '../Maps/'
osmFile = 'linkoping.osm'
figFile = 'linkoping.png'
osmMap = loadOSMmap(dataDir + osmFile, dataDir + figFile)

num_nodes = len(osmMap.nodes)  # Number of nodes in the map




def f_next(x):
    """Compute, neighbours for a node"""
    cx = osmMap.distancematrix[x, :].tocoo()
    return cx.col, np.full(cx.col.shape, np.nan), cx.data


# # Display basic information about map

# Print some basic map information


osmMap.info()


# Plot the map

plt.figure(10)
osmMap.plotmap()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Linköping');


# Which nodes are neighbors to node with index 110?


n, _, d = f_next(110)
print(f'Neighbours: {n}')
print(f'Distances: {d}')


# Look up the distance (in meters) between the nodes 110 and 3400 in the distance matrix?


print(osmMap.distancematrix[110, 3400])


# Latitude and longitude of node 110


p = osmMap.nodeposition[110]
print(f'Longitude = {p[0]:.3f}, Latitude = {p[1]:.3f}')


# Plot the distance matrix and illustrate sparseness


plt.figure(20)
plt.spy(osmMap.distancematrix>0, markersize=0.5)
plt.xlabel('Node index')
plt.ylabel('Node index')
density = np.sum(osmMap.distancematrix>0)/num_nodes**2
_ = plt.title(f'Density {density*100:.2f}%')


# # Define planning mission

# Some pre-defined missions to experiment with. To use the first pre-defined mission, call the planner with
# ```planner(num_nodes, pre_mission[0], f_next, cost_to_go)```.


pre_mission = [
    {'start': {'id': 10906}, 'goal': {'id': 1024}},
    {'start': {'id': 3987}, 'goal': {'id': 4724}},
    {'start': {'id': 423}, 'goal': {'id': 5119}}]
mission = pre_mission[0]


# In the map, click on start and goal positions to define a mission. Try different missions, ranging from easy to more complex. 
# 
# An easy mission is a mission in the city centre; while a more difficult could be from Vallastaden to Tannefors.


""" plt.figure(30, clear=True)
osmMap.plotmap()
plt.title('Linköping - click in map to define mission')
mission = {}

mission['start'] = osmMap.getmapposition()
plt.plot(mission['start']['pos'][0], mission['start']['pos'][1], 'bx')
mission['goal'] = osmMap.getmapposition()
plt.plot(mission['goal']['pos'][0], mission['goal']['pos'][1], 'bx')

plt.xlabel('Longitude')
plt.ylabel('Latitude')

print('Mission: Go from node %d ' % (mission['start']['id']), end='')
if mission['start']['name'] != '':
    print('(' + mission['start']['name'] + ')', end='')
print(' to node %d ' % (mission['goal']['id']), end='')
if mission['goal']['name'] != '':
    print('(' + mission['goal']['name'] + ')', end='')
print('')

mission
 """

# # Implement planners
if flag_planner == 1 or flag_planner ==15:

    def DepthFirst(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """Depth first planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)

        startNode = mission['start']['id']
        goalNode = mission['goal']['id']

        q = LIFO()
        q.insert(startNode)
        foundPlan = False

        while not q.IsEmpty():
            x = q.pop()
            if x == goalNode:
                foundPlan = True
                break
            neighbours, u, d = f_next(x)
            for xi, ui, di in zip(neighbours, u, d):
                if previous[xi] == unvis_node:
                    previous[xi] = x
                    q.insert(xi)
                    cost_to_come[xi] = cost_to_come[x] + di
                    if num_controls > 0:
                        control_to_come[xi] = ui

        # Recreate the plan by traversing previous from goal node
        if not foundPlan:
            return []
        else:
            plan = [goalNode]
            length = cost_to_come[goalNode]
            control = []
            while plan[0] != startNode:
                if num_controls > 0:
                    control.insert(0, control_to_come[plan[0]])
                plan.insert(0, previous[plan[0]])

            return {'plan': plan,
                    'length': length,
                    'num_visited_nodes': np.sum(previous != unvis_node),
                    'name': 'DepthFirst',
                    'time': t.toc(),
                    'control': control,
                    'visited_nodes': previous[previous != unvis_node]}


    # ## Planning example using the DepthFirst planner

    # Make a plan using the ```DepthFirst``` planner


    df_plan = DepthFirst(num_nodes, mission, f_next)
    print("..................DFS................................................")
    print('{:.1f} m, {} visited nodes, planning time {:.1f} msek'.format(
        df_plan['length'],
        df_plan['num_visited_nodes'],
        df_plan['time']*1e3))


    # Plot the resulting plan


    plt.figure(40, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(df_plan['plan'], 'b',
                    label=f"Depth first ({df_plan['length']:.1f} m)")
    plt.title('Linköping')
    _ = plt.legend()


    # Plot nodes visited during search


    plt.figure(41, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(df_plan['visited_nodes'], 'b.')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    _ = plt.title('Nodes visited during DepthFirst search')


    # Names of roads along the plan ...


    planWayNames = osmMap.getplanwaynames(df_plan['plan'])
    print('Start: ', end='')
    for w in planWayNames[:-1]:
        print(w + ' -- ', end='')
    print('Goal: ' + planWayNames[-1])
    #plt.show()




#####################################################################################################################################################
# Here, write your code for your planners. Start with the template code for the breadth first search and extend.############################################

if flag_planner == 2 or flag_planner ==15:

    def BreadthFirst(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """Breadth first planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)

        startNode = mission['start']['id']
        goalNode = mission['goal']['id']

        q = FIFO()
        q.insert(startNode)
        foundPlan = False

        while not q.IsEmpty():
            x = q.pop()
            if x == goalNode:
                foundPlan = True
                break
            neighbours, u, d = f_next(x)
            for xi, ui, di in zip(neighbours, u, d):
                if previous[xi] == unvis_node:
                    previous[xi] = x
                    q.insert(xi)
                    cost_to_come[xi] = cost_to_come[x] + di
                    if num_controls > 0:
                        control_to_come[xi] = ui

        # Recreate the plan by traversing previous from goal node
        if not foundPlan:
            return []
        else:
            plan = [goalNode]
            length = cost_to_come[goalNode]
            control = []
            while plan[0] != startNode:
                if num_controls > 0:
                    control.insert(0, control_to_come[plan[0]])
                plan.insert(0, previous[plan[0]])

            return {'plan': plan,
                    'length': length,
                    'num_visited_nodes': np.sum(previous != unvis_node),
                    'name': 'BreadthFirst',
                    'time': t.toc(),
                    'control': control,
                    'visited_nodes': previous[previous != unvis_node]}


    # Make a plan using the ```BreadthFirst``` planner


    bf_plan = BreadthFirst(num_nodes, mission, f_next)
    print("..................BFS................................................")
    print('{:.1f} m, {} visited nodes, planning time {:.1f} msek'.format(
        bf_plan['length'],
        bf_plan['num_visited_nodes'],
        bf_plan['time']*1e3))


    # Plot the resulting plan


    plt.figure(50, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(bf_plan['plan'], 'b',
                    label=f"Breadth first ({bf_plan['length']:.1f} m)")
    plt.title('Linköping')
    _ = plt.legend()


    # Plot nodes visited during search


    plt.figure(51, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(bf_plan['visited_nodes'], 'b.')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    _ = plt.title('Nodes visited during BreadthFirst search')


    # Names of roads along the plan ...


    planWayNames = osmMap.getplanwaynames(bf_plan['plan'])
    print('Start: ', end='')
    for w in planWayNames[:-1]:
        print(w + ' -- ', end='')
    print('Goal: ' + planWayNames[-1])
    #plt.show()


########################################################################################################################################################

if flag_planner == 3 or flag_planner ==15:

    def Dijkstra(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """Dijkstra  planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)

        startNode = mission['start']['id']
        goalNode = mission['goal']['id']

        q = PriorityQueue()
        q.insert(cost_to_come[startNode], startNode)
        foundPlan = False

        while not q.IsEmpty():
            _ ,x = q.pop()
            if x == goalNode:
                foundPlan = True
                break
            neighbours, u, d = f_next(x)
            for xi, ui, di in zip(neighbours, u, d):
                if (previous[xi] == unvis_node) or (cost_to_come[xi] > cost_to_come[x] + di) :
                    previous[xi] = x
                    cost_to_come[xi] = cost_to_come[x] + di
                    q.insert(cost_to_come[xi], xi)
                    if num_controls > 0:
                        control_to_come[xi] = ui

        # Recreate the plan by traversing previous from goal node
        if not foundPlan:
            return []
        else:
            plan = [goalNode]
            length = cost_to_come[goalNode]
            control = []
            while plan[0] != startNode:
                if num_controls > 0:
                    control.insert(0, control_to_come[plan[0]])
                plan.insert(0, previous[plan[0]])

            return {'plan': plan,
                    'length': length,
                    'num_visited_nodes': np.sum(previous != unvis_node),
                    'name': 'Dijkstra',
                    'time': t.toc(),
                    'control': control,
                    'visited_nodes': previous[previous != unvis_node]}


    ds_plan = Dijkstra(num_nodes, mission, f_next)
    print("..................Dijkstra................................................")
    print('{:.1f} m, {} visited nodes, planning time {:.1f} msek'.format(
        ds_plan['length'],
        ds_plan['num_visited_nodes'],
        ds_plan['time']*1e3))


    # Plot the resulting plan


    plt.figure(60, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(ds_plan['plan'], 'b',
                    label=f"Dijkstra ({ds_plan['length']:.1f} m)")
    plt.title('Linköping')
    _ = plt.legend()


    # Plot nodes visited during search


    plt.figure(61, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(ds_plan['visited_nodes'], 'b.')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    _ = plt.title('Nodes visited during Dijkstra search')


    # Names of roads along the plan ...


    planWayNames = osmMap.getplanwaynames(ds_plan['plan'])
    print('Start: ', end='')
    for w in planWayNames[:-1]:
        print(w + ' -- ', end='')
    print('Goal: ' + planWayNames[-1])
    #plt.show()


################################################################################################################################

# Define the heuristic for Astar and BestFirst. The ```LatLongDistance``` function will be useful.


def cost_to_go(x, xg):
    p_x = osmMap.nodeposition[x]
    p_g = osmMap.nodeposition[xg]
    return p_x[0]-p_g[0];

#################################################################################################################################

if flag_planner == 4 or flag_planner ==15:

    def Astar(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """A Star  planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)

        startNode = mission['start']['id']
        goalNode = mission['goal']['id']

        heuristic = cost_to_go(startNode, goalNode)

        q = PriorityQueue()
        q.insert(cost_to_come[startNode]+heuristic, startNode)
        foundPlan = False
    
        while not q.IsEmpty():
            _ ,x = q.pop()
            if x == goalNode:
                foundPlan = True
                break
            neighbours, u, d = f_next(x)
            for xi, ui, di in zip(neighbours, u, d):
                if (previous[xi] == unvis_node) or (cost_to_come[xi] > cost_to_come[x] + di) :
                    previous[xi] = x
                    cost_to_come[xi] = cost_to_come[x] + di
                    q.insert(cost_to_come[xi] + cost_to_go(xi, goalNode), xi)
                    if num_controls > 0:
                        control_to_come[xi] = ui

        # Recreate the plan by traversing previous from goal node
        if not foundPlan:
            return []
        else:
            plan = [goalNode]
            length = cost_to_come[goalNode]
            control = []
            while plan[0] != startNode:
                if num_controls > 0:
                    control.insert(0, control_to_come[plan[0]])
                plan.insert(0, previous[plan[0]])

            return {'plan': plan,
                    'length': length,
                    'num_visited_nodes': np.sum(previous != unvis_node),
                    'name': 'AStar',
                    'time': t.toc(),
                    'control': control,
                    'visited_nodes': previous[previous != unvis_node]}


    as_plan = Astar(num_nodes, mission, f_next)
    print("..................A Star................................................")
    print('{:.1f} m, {} visited nodes, planning time {:.1f} msek'.format(
        as_plan['length'],
        as_plan['num_visited_nodes'],
        as_plan['time']*1e3))


    # Plot the resulting plan


    plt.figure(70, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(as_plan['plan'], 'b',
                    label=f"AStar ({as_plan['length']:.1f} m)")
    plt.title('Linköping')
    _ = plt.legend()


    # Plot nodes visited during search


    plt.figure(71, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(as_plan['visited_nodes'], 'b.')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    _ = plt.title('Nodes visited during AStar search')


    # Names of roads along the plan ...


    planWayNames = osmMap.getplanwaynames(as_plan['plan'])
    print('Start: ', end='')
    for w in planWayNames[:-1]:
        print(w + ' -- ', end='')
    print('Goal: ' + planWayNames[-1])
    #plt.show()


#################################################################################################################################

if flag_planner == 5 or flag_planner ==15:

    def BestFirst(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """Best First  planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int)

        startNode = mission['start']['id']
        goalNode = mission['goal']['id']

        heuristic = cost_to_go(startNode, goalNode)


        q = PriorityQueue()
        q.insert(heuristic, startNode)
        foundPlan = False

        while not q.IsEmpty():
            _ ,x = q.pop()
            if x == goalNode:
                foundPlan = True
                break
            neighbours, u, d = f_next(x)
            for xi, ui, di in zip(neighbours, u, d):
                if (previous[xi] == unvis_node):
                    previous[xi] = x
                    cost_to_come[xi] = cost_to_come[x] + di
                    q.insert(cost_to_go(xi, goalNode), xi)
                    if num_controls > 0:
                        control_to_come[xi] = ui

        # Recreate the plan by traversing previous from goal node
        if not foundPlan:
            return []
        else:
            plan = [goalNode]
            length = cost_to_come[goalNode]
            control = []
            while plan[0] != startNode:
                if num_controls > 0:
                    control.insert(0, control_to_come[plan[0]])
                plan.insert(0, previous[plan[0]])

            return {'plan': plan,
                    'length': length,
                    'num_visited_nodes': np.sum(previous != unvis_node),
                    'name': 'BestFirst',
                    'time': t.toc(),
                    'control': control,
                    'visited_nodes': previous[previous != unvis_node]}

    # Make a plan using the ```BreadthFirst``` planner


    bs_plan = BestFirst(num_nodes, mission, f_next)
    print("..................BestFirst................................................")
    print('{:.1f} m, {} visited nodes, planning time {:.1f} msek'.format(
        bs_plan['length'],
        bs_plan['num_visited_nodes'],
        bs_plan['time']*1e3))


    # Plot the resulting plan


    plt.figure(80, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(bs_plan['plan'], 'b',
                    label=f"BestFirst ({bs_plan['length']:.1f} m)")
    plt.title('Linköping')
    _ = plt.legend()


    # Plot nodes visited during search


    plt.figure(81, clear=True)
    osmMap.plotmap()
    osmMap.plotplan(bs_plan['visited_nodes'], 'b.')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    _ = plt.title('Nodes visited during BestFirst search')


    # Names of roads along the plan ...


    planWayNames = osmMap.getplanwaynames(bs_plan['plan'])
    print('Start: ', end='')
    for w in planWayNames[:-1]:
        print(w + ' -- ', end='')
    print('Goal: ' + planWayNames[-1])
    #plt.show()


#################################################################################################################################


# # Define heuristic for Astar and BestFirst planners



# # Investigations using all planners

""" res_dijkstra = Dijkstra(num_nodes, pre_mission[0], f_next)
res_astar = Astar(num_nodes, pre_mission[0], f_next, cost_to_go)
assert res_dijkstra['length'] == res_astar['length']
assert abs(res_astar['length'] - 5085.5957) < 1e-2 


res_dijkstra = Dijkstra(num_nodes, pre_mission[1], f_next)
res_astar = Astar(num_nodes, pre_mission[1], f_next, cost_to_go)
assert res_dijkstra['length'] == res_astar['length']
assert abs(res_astar['length'] - 2646.2140) < 1e-2  

res_dijkstra = Dijkstra(num_nodes, pre_mission[2], f_next)
res_astar = Astar(num_nodes, pre_mission[2], f_next, cost_to_go)
assert res_dijkstra['length'] == res_astar['length']
assert abs(res_astar['length'] - 1860.7143) < 1e-2 """

plt.show()
print("done")


