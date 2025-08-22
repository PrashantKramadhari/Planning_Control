import matplotlib.pyplot as plt
import numpy as np

# Auxiliary function used in the value-iteration loop to compute the value
# function
def V_update(row, col, V, params):
    """ Update value function
    
       v, r = V_update(row, col, V, params):
       
        Input:
           row, col - state
           V - current value function
           params - parameters

        Output:
           v - new value in value function for current state
           r - reward
    """
    def get_V_next(V, row, col, a, params):
        occ_grid = params['occ_grid']
        n_rows = params['n_rows']
        n_cols = params['n_cols']
        R = params['R']

        V_next = V[row, col]
        r = R[row, col]

        # Move left
        if a == 0:
            row_next = row
            col_next = col - 1
            if col_next >= 0 and occ_grid[row_next, col_next] == 0:
                V_next = V[row_next, col_next]
                r = R[row_next, col_next]

        # Move right
        if a == 1:
            row_next = row
            col_next = col + 1
            if col_next < n_cols and occ_grid[row_next, col_next] == 0:
                V_next = V[row_next, col_next]
                r = R[row_next, col_next]

        # Move up
        if a == 2:
            row_next = row - 1
            col_next = col
            if row_next >= 0 and occ_grid[row_next, col_next] == 0:
                V_next = V[row_next, col_next]
                r = R[row_next, col_next]

        # Move down
        if a == 3:
            row_next = row + 1
            col_next = col
            if row_next < n_rows and occ_grid[row_next, col_next] == 0:
                V_next = V[row_next, col_next]
                r = R[row_next, col_next]
        return V_next, r
    
    P_move_action = params['P_move_action']
    P_dist = params['P_dist']
    gamma = params['gamma']

    q = np.zeros(4)
    
    # Iterate over all possible actions

    # Move left
    V_next1, r1 = get_V_next(V, row, col, 0, params)
    V_next2, r2 = get_V_next(V, row, col, 2, params)
    V_next3, r3 = get_V_next(V, row, col, 3, params)

    q[0] = (P_move_action*(r1 + gamma*V_next1) + 
            P_dist*(r2 + gamma*V_next2) + 
            P_dist*(r3 + gamma*V_next3))

    # Move right
    V_next1, r1 = get_V_next(V, row, col, 1, params)
    V_next2, r2 = get_V_next(V, row, col, 2, params)
    V_next3, r3 = get_V_next(V, row, col, 3, params)

    q[1] = (P_move_action*(r1 + gamma*V_next1) + 
            P_dist*(r2 + gamma*V_next2) + 
            P_dist*(r3 + gamma*V_next3))

    # Move up
    V_next1, r1 = get_V_next(V, row, col, 2, params)
    V_next2, r2 = get_V_next(V, row, col, 0, params)
    V_next3, r3 = get_V_next(V, row, col, 1, params)

    q[2] = (P_move_action*(r1 + gamma*V_next1) + 
            P_dist*(r2 + gamma*V_next2) + 
            P_dist*(r3 + gamma*V_next3))

    # Move down
    V_next1, r1 = get_V_next(V, row, col, 3, params)
    V_next2, r2 = get_V_next(V, row, col, 0, params)
    V_next3, r3 = get_V_next(V, row, col, 1, params)

    q[3] = (P_move_action*(r1 + gamma*V_next1) + 
            P_dist*(r2 + gamma*V_next2) +
            P_dist*(r3 + gamma*V_next3))

    max_a = np.argmax(q)
            
    return q[max_a], max_a

def plot_iter(V, Pi, params):
    """Plot current state of value function and policy."""
    n_rows = params['n_rows']
    n_cols = params['n_cols']    
    occ_grid = params['occ_grid']
    R = params['R']

    goal = params['goal']
    sink = params['sink']

    actions = ['left','right','up','down']

    fig1 = plt.figure(1, clear=True)
    for row in range(n_rows):
        for col in range(n_cols):
            if occ_grid[row, col] == 1:
                plt.text(col, n_rows - 1 - row, '0.0', color='k', ha='center', va='center')
            elif np.any(np.logical_and(row==sink[:, 0], col==sink[:, 1])):
                plt.text(col, n_rows - 1 - row, "{:.3f}".format(R[row, col]), 
                         color='r', ha='center', va='center')
            elif np.all([row, col]==goal):
                plt.text(col, n_rows - 1 - row, "{:.3f}".format(R[row, col]), 
                         color='g', ha='center', va='center')
            else:
                plt.text(col, n_rows - 1 - row, "{:.3f}".format(V[row, col]), 
                         color='b', ha='center', va='center')
    plt.axis([-1, n_cols, -1, n_rows])
    plt.axis('off')


    fig2 = plt.figure(2, clear=True)
    for row in range(n_rows):
        for col in range(n_cols):
            if not Pi[row, col] == -1:
                plt.text(col, n_rows - 1 - row, actions[Pi[row, col]], 
                         color='k', ha='center', va='center')
            elif np.all([row, col]==goal):
                plt.text(col, n_rows - 1 - row, "{:.3f}".format(R[row, col]), 
                         color='g', ha='center', va='center')
            elif np.any(np.logical_and(row==sink[:, 0], col==sink[:, 1])):
                plt.text(col, n_rows - 1 - row, "{:.3f}".format(R[row, col]), 
                         color='r', ha='center', va='center')
    plt.axis([-1, n_cols, -1, n_rows])
    plt.axis('off')

    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig2.canvas.draw()
    fig2.canvas.flush_events()

def next_state(s_curr, action, params):
    """Get next state and reward
    
        s, r = next_state(s_curr, action, params)
        
        Input:
            s_curr - current state
            action - intended action
            prams - parameters
        
        Output:
            s - next state
            r - reward
    """
    P_dist = params['P_dist']
    R = params['R']
    n_rows = params['n_rows']
    n_cols = params['n_cols']
    occ_grid = params['occ_grid']

    rnd = np.random.uniform()

    s_next = s_curr

    # Actions - ['left','right','up','down']

    if rnd <= P_dist:
        if action == 0:
            move = 2
        elif action == 1:
            move = 2
        elif action == 2:
            move = 1
        else:
            move = 0
    elif rnd < 2*P_dist:
        if action == 0:
            move = 3
        elif action == 1:
            move = 3
        elif action == 2:
            move = 1
        else:
            move = 1
    else:
        move = action

    # Move left
    if move == 0:
        row_next = s_curr[0]
        col_next = s_curr[1] - 1
        if col_next >= 0 and occ_grid[row_next, col_next] == 0:
            s_next = [row_next, col_next]

    # Move right
    if move == 1:
        row_next = s_curr[0]
        col_next = s_curr[1] + 1
        if col_next < n_cols and occ_grid[row_next, col_next] == 0:
            s_next = [row_next, col_next]

    # Move up
    if move == 2:
        row_next = s_curr[0] - 1
        col_next = s_curr[1]
        if row_next >= 0 and occ_grid[row_next, col_next] == 0:
            s_next = [row_next, col_next]

    # Move down
    if move == 3:
        row_next = s_curr[0] + 1
        col_next = s_curr[1]
        if row_next < n_rows and occ_grid[row_next, col_next] == 0:
            s_next = [row_next, col_next]

    r = R[s_next[0], s_next[1]]
    return s_next, r

def BoxOff(*argin):
    if len(argin)>0:
        ax=argin[0]
    else:
        ax=plt.gca();
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

