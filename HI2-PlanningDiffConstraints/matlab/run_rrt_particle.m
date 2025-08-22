%% Run the RRT for a particle moving in a plane (2D world)

clear;
close all;
addpath Functions

%% Define world with obstacles

world = BoxWorld({[0, 10], [0, 10]});

world.add_box(2, 2, 6, 6)
world.add_box(1, 6, 4, 3)
world.add_box(4, 1, 5, 4)

figure(10)
clf()
world.draw()
axis([world.xmin, world.xmax, world.ymin, world.ymax])
xlabel('x');
ylabel('y');

start = [1; 1];     % Start state
goal = [9; 9];      % Goal state

% Define parameters and data structures

opts.beta = 0.05; % Probability for selecting goal state as target state 
                  % in the sample
opts.delta = 0.1; % Step size
opts.eps = -0.01; % Threshold for stopping the search (negative for full search)
opts.K = 5000;    % Maximum number of iterations, if eps < 0

%% Solve problem

fprintf('Planning ...\n');
[goal_idx, nodes, parents, T] = rrt_particle(start, goal, world, opts);
fprintf('Finished in %.2f sek\n', T);

%% YOUR CODE HERE


