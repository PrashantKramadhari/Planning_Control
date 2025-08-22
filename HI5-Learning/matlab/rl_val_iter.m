% Value iteration for solving an extension of Example 6.6 in
% Sutton, R. S., & A. G. Barto: Reinforcement learning: An introduction.
% MIT Press, 2018.

clear;
close all

gamma = 0.99; % discount factor
R_goal = 0.0; % reward for reaching goal state
R_sink = -10.0; % reward for reaching 'cliff' states
R_grid = -0.1; % reward for remaining states

P_move_action = 0.99; % probability of moving in the direction specified by action
P_dist = (1-P_move_action)/2; % probability of moving sideways compared to intended
                       % because of disturbance

n_rows = 4;
n_cols = 5;

goal = [4 5]; % element index goal state
sink = [4 2; 4 3; 4 4]; % element indices for cliff states

% Setup reward matrix R
R = R_grid*ones(n_rows,n_cols);
R(goal(1),goal(2)) = R_goal;
R(sink(:,1),sink(:,2)) = R_sink;

% Occupancy grid defines states where there are obstacles
occ_grid = zeros(n_rows,n_cols);
occ_grid(2,2) = 1;

% Save parameters in a struct
params.gamma = gamma;
params.R_goal = R_goal;
params.R_sink = R_sink;
params.R_grid = R_grid;
params.P_move_action = P_move_action;
params.P_dist = P_dist;
params.n_rows = n_rows;
params.n_cols = n_cols;
params.goal = goal;
params.sink = sink;
params.R = R;
params.occ_grid = occ_grid;

% Initilaize value function for each state
V = zeros(n_rows,n_cols);

% Actions - ['left','right','up','down'] counted as 1-4

% Initialize vector for policy
Pi = -1*ones(n_rows,n_cols);

% Main loop for value iteration
% Algorithm according to Section 4.4 in Sutton, R. S., & A. G. Barto: 
% Reinforcement learning: An introduction. MIT Press, 2018.

converged = false;

while ~converged
    
    Delta = 0;
    
    for i = 1:n_rows
        for j = 1:n_cols
            if (occ_grid(i,j) == 1) || (i == goal(1) && j == goal(2)) || ...
                    (i == sink(1,1) && j == sink(1,2)) || ...
                    (i == sink(2,1) && j == sink(2,2)) || ...
                    (i == sink(3,1) && j == sink(3,2))
                continue;
            end
            v = V(i,j);
            [V(i,j), max_a] = V_update(i,j,V,params);
            Pi(i,j) = max_a;
            Delta = max(Delta,abs(v-V(i,j)));
        end
    end
    
    % Visualize current value function and associated actions according to
    % current policy
    plot_iter(V,Pi,params);
    
    V
    Pi
    pause;
    
    if Delta < 1e-6
        converged = true;
    end
    
end

% Auxiliary function to plot the current iteration of value function and
% policy
function plot_iter(V,Pi,params)

n_rows = params.n_rows;
n_cols = params.n_cols;
occ_grid = params.occ_grid;
R = params.R;
goal = params.goal;
sink = params.sink;

actions = {'left','right','up','down'};

figure(1)
clf;
hold on;
for i = 1:n_rows
    for j = 1:n_cols
        if occ_grid(i,j) == 1
            text(j-0.25,n_rows-i+1,sprintf('%.3f',0.0),'fontsize',15,'color','k')
        elseif i == sink(1,1) && j == sink(1,2)
            text(j-0.25,n_rows-i+1,sprintf('%.3f',R(i,j)),'fontsize',15,'color','r')
        elseif i == sink(2,1) && j == sink(2,2)
            text(j-0.25,n_rows-i+1,sprintf('%.3f',R(i,j)),'fontsize',15,'color','r')
        elseif i == sink(3,1) && j == sink(3,2)
            text(j-0.25,n_rows-i+1,sprintf('%.3f',R(i,j)),'fontsize',15,'color','r')
        elseif i == goal(1) && j == goal(2)
            text(j-0.25,n_rows-i+1,sprintf('%.3f',R(i,j)),'fontsize',15,'color','g')
        else
            text(j-0.25,n_rows-i+1,sprintf('%.3f',V(i,j)),'fontsize',15,'color','b')
        end
    end
end
axis([0 n_cols+1 0 n_rows+1])
axis off

figure(2)

clf;
hold on;
for i = 1:n_rows
    for j = 1:n_cols
        if Pi(i,j) ~= -1
            text(j-0.25,n_rows-i+1,actions(Pi(i,j)),'fontsize',15)
        elseif i == goal(1) && j == goal(2)
            text(j-0.25,n_rows-i+1,sprintf('%.3f',R(i,j)),'fontsize',15,'color','g')
        elseif i == sink(1,1) && j == sink(1,2)
            text(j-0.25,n_rows-i+1,sprintf('%.3f',R(i,j)),'fontsize',15,'color','r')
        elseif i == sink(2,1) && j == sink(2,2)
            text(j-0.25,n_rows-i+1,sprintf('%.3f',R(i,j)),'fontsize',15,'color','r')
        elseif i == sink(3,1) && j == sink(3,2)
            text(j-0.25,n_rows-i+1,sprintf('%.3f',R(i,j)),'fontsize',15,'color','r')
        end
    end
end
axis([0 n_cols+1 0 n_rows+1])
axis off

end

% Auxiliary function used in the value-iteration loop to compute the 
% update of the value function
function [V_val,max_a] = V_update(i,j,V,params)

P_move_action = params.P_move_action;
P_dist = params.P_dist;
gamma = params.gamma;


% Iterate over all possible actions

% Move left
[V_next1,r1] = get_V_next(V,i,j,1,params);
[V_next2,r2] = get_V_next(V,i,j,3,params);
[V_next3,r3] = get_V_next(V,i,j,4,params);

q(1) = P_move_action*(r1+gamma*V_next1) + ...
    P_dist*(r2+gamma*V_next2) + ...
    P_dist*(r3+gamma*V_next3);

% Move right
[V_next1,r1] = get_V_next(V,i,j,2,params);
[V_next2,r2] = get_V_next(V,i,j,3,params);
[V_next3,r3] = get_V_next(V,i,j,4,params);

q(2) = P_move_action*(r1+gamma*V_next1) + ...
    P_dist*(r2+gamma*V_next2) + ...
    P_dist*(r3+gamma*V_next3);

% Move up
[V_next1,r1] = get_V_next(V,i,j,3,params);
[V_next2,r2] = get_V_next(V,i,j,1,params);
[V_next3,r3] = get_V_next(V,i,j,2,params);

q(3) = P_move_action*(r1+gamma*V_next1) + ...
    P_dist*(r2+gamma*V_next2) + ...
    P_dist*(r3+gamma*V_next3);

% Move down
[V_next1,r1] = get_V_next(V,i,j,4,params);
[V_next2,r2] = get_V_next(V,i,j,1,params);
[V_next3,r3] = get_V_next(V,i,j,2,params);

q(4) = P_move_action*(r1+gamma*V_next1) + ...
    P_dist*(r2+gamma*V_next2) + ...
    P_dist*(r3+gamma*V_next3);

[V_val,max_a] = max(q);

end

% Auxiliary function to function V_update for computing the value function 
% at the next state. If next state is outside of world, we stay in the 
% current state and receive the corresponding reward again.
function [V_next,r] = get_V_next(V,i,j,a,params)

occ_grid = params.occ_grid;
n_rows = params.n_rows;
n_cols = params.n_cols;
R = params.R;

V_next = V(i,j);
r = R(i,j);

% Move left
if a == 1
    i_next = i;
    j_next = j-1;
    if j_next > 0 && occ_grid(i_next,j_next) == 0
        V_next = V(i_next,j_next);
        r = R(i_next,j_next);
    end
end

% Move right
if a == 2
    i_next = i;
    j_next = j+1;
    if j_next <= n_cols && occ_grid(i_next,j_next) == 0
        V_next = V(i_next,j_next);
        r = R(i_next,j_next);
    end
end

% Move up
if a == 3
    i_next = i-1;
    j_next = j;
    if i_next > 0 && occ_grid(i_next,j_next) == 0
        V_next = V(i_next,j_next);
        r = R(i_next,j_next);
    end
end

% Move down
if a == 4
    i_next = i+1;
    j_next = j;
    if i_next <= n_rows && occ_grid(i_next,j_next) == 0
        V_next = V(i_next,j_next);
        r = R(i_next,j_next);
    end
end

end