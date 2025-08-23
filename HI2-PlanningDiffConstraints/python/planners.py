from queues import FIFO, LIFO, PriorityQueue
import numpy as np
from misc import Timer


def depth_first(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """Depth first planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int32, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int32)

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

def breadth_first(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """Breadth first planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int32, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int32)

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

def dijkstra(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """Dijkstra  planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int32, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int32)

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

def astar(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """A Star  planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int32, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int32)

        startNode = mission['start']['id']
        goalNode = mission['goal']['id']

        heur = heuristic(startNode, goalNode)

        q = PriorityQueue()
        q.insert(cost_to_come[startNode]+heur, startNode)
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
                    q.insert(cost_to_come[xi] + heuristic(xi, goalNode), xi)
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

def best_first(num_nodes, mission, f_next, heuristic=None, num_controls=0):
        """Best First  planner."""
        t = Timer()
        t.tic()
        
        unvis_node = -1
        previous = np.full(num_nodes, dtype=np.int32, fill_value=unvis_node)
        cost_to_come = np.zeros(num_nodes)
        control_to_come = np.zeros((num_nodes, num_controls), dtype=np.int32)

        startNode = mission['start']['id']
        goalNode = mission['goal']['id']

        heuri = heuristic(startNode, goalNode)


        q = PriorityQueue()
        q.insert(heuri, startNode)
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
                    q.insert(heuristic(xi, goalNode), xi)
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
