% Q-learning for solving an extension of Example 6.6 in
% Sutton, R. S., & A. G. Barto: Reinforcement learning: An introduction.
% MIT Press, 2018.

clear;
close all;

gamma = 0.99; % discount factor
R_goal = 0.0; % reward for reaching goal state
R_sink = -10.0; % reward for reaching 'cliff' states
R_grid = -0.1; % reward for remaining states

alpha = 0.5; % learning rate in Q-update
eps = 0.5; % epsilon-greedy parameter

P_move_action = 1.0; % probability of moving in the direction specified by action
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
params.alpha = alpha;
params.eps = eps;
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

% Initialize cell object for Q function with random values
for i = 1:n_rows
    for j = 1:n_cols
        Q{i,j} = rand(4,1);
    end
end

% Initialize Q for terminal states to zero
Q{goal(1),goal(2)} = zeros(4,1);
Q{sink(1,1),sink(1,2)} = zeros(4,1);
Q{sink(2,1),sink(2,2)} = zeros(4,1);
Q{sink(3,1),sink(3,2)} = zeros(4,1);

% Initialize vector for policy
Pi = -1*ones(n_rows,n_cols);

% Define number of iterations for Q-learning
nbr_iters = 10000;

% Initialize vector for sum of rewards for each episode
sum_r = zeros(nbr_iters,1);

% Main loop for Q-learning
% Algorithm according to Section 6.5 in Sutton, R. S., & A. G. Barto: 
% Reinforcement learning: An introduction. MIT Press, 2018.

k = 1;
while k < nbr_iters
    % Start state
    s_curr = [n_rows,1];
    
    terminal_state = false;
    
    while ~terminal_state
        % Select action according to epsilon-greedy strategy
        action = select_eps_greedy(s_curr,k,Q,params);
        
        % Perform the action and receive reward and next state
        [s_next,r] = next_state(s_curr,action,params);
        
        % Q-learning update of action-value function
        Q{s_curr(1),s_curr(2)}(action) = Q{s_curr(1),s_curr(2)}(action) + ...
            alpha*(r + gamma*max(Q{s_next(1),s_next(2)}) - ...
            Q{s_curr(1),s_curr(2)}(action));
        
        % Update the sum of reward vector
        sum_r(k) = sum_r(k) + r;
        
        s_curr = s_next;
        
        % Check if a terminal state has been reached (closes an episode)
        if (s_curr(1) == goal(1) && s_curr(2) == goal(2)) || ...
                (s_curr(1) == sink(1,1) && s_curr(2) == sink(1,2)) || ...
                (s_curr(1) == sink(2,1) && s_curr(2) == sink(2,2)) || ...
                (s_curr(1) == sink(3,1) && s_curr(2) == sink(3,2))
            terminal_state = true;
            
            % Update value function and policy
            for i = 1:n_rows
                for j = 1:n_cols
                    if (occ_grid(i,j) == 1) || (i == goal(1) && j == goal(2)) || ...
                            (i == sink(1,1) && j == sink(1,2)) || ...
                            (i == sink(2,1) && j == sink(2,2)) || ...
                            (i == sink(3,1) && j == sink(3,2))
                        continue;
                    end
                    [V_ij,max_a] = max(Q{i,j});
                    V(i,j) = V_ij;
                    Pi(i,j) = max_a;
                end
            end
            
            V
            Pi
            
        end
    end
    k = k+1;
end

% Visualize the value function and policy after all iterations
plot_iter(V,Pi,params);

% Compute average of reward for N episodes for smoothing
N = 40;
for i = N:length(sum_r)
    mean_sum_r(i) = mean(sum_r(i-N+1:i));
end

% Visualize the evolution of the reward for each episode
figure(3)
plot(N:nbr_iters,mean_sum_r(N:end))
grid on
title('Sum of rewards for each episode (average over 40)')

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

% Select the action to take using an epsilon-greedy strategy
function action = select_eps_greedy(s_curr,k,Q,params)

eps = params.eps;

rnd = rand(1);
[~,max_a] = max(Q{s_curr(1),s_curr(2)});

a_list = [];

for i = 1:4
    if i ~= max_a
        a_list(end+1) = i;
    end
end

if rnd < 1-eps+eps/4
    action = max_a;
elseif rnd < 1-eps+eps/2
    action = a_list(1);
elseif rnd < 1-eps+3*eps/4
    action = a_list(2);
else
    action = a_list(3);
end

end


function [s_next,r] = next_state(s_curr,action,params)

P_dist = params.P_dist;
R = params.R;
n_rows = params.n_rows;
n_cols = params.n_cols;
occ_grid = params.occ_grid;

rnd = rand(1);

s_next = s_curr;

% Actions - ['left','right','up','down']

if rnd <= P_dist
    if action == 1
        move = 3;
    elseif action == 2
        move = 3;
    elseif action == 3
        move = 1;
    else
        move = 1;
    end
elseif rnd < 2*P_dist
    if action == 1
        move = 4;
    elseif action == 2
        move = 4;
    elseif action == 3
        move = 2;
    else
        move = 2;
    end
else
    move = action;
end

% Move left
if move == 1
    i_next = s_curr(1);
    j_next = s_curr(2)-1;
    if j_next > 0 && occ_grid(i_next,j_next) == 0
        s_next = [i_next,j_next];
    end
end

% Move right
if move == 2
    i_next = s_curr(1);
    j_next = s_curr(2)+1;
    if j_next <= n_cols && occ_grid(i_next,j_next) == 0
        s_next = [i_next,j_next];
    end
end

% Move up
if move == 3
    i_next = s_curr(1)-1;
    j_next = s_curr(2);
    if i_next > 0 && occ_grid(i_next,j_next) == 0
        s_next = [i_next,j_next];
    end
end

% Move down
if move == 4
    i_next = s_curr(1)+1;
    j_next = s_curr(2);
    if i_next <= n_rows && occ_grid(i_next,j_next) == 0
        s_next = [i_next,j_next];
    end
end

r = R(s_next(1),s_next(2));

end