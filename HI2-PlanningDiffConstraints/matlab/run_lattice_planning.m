clear
close all
addpath Functions
% If you are in the student labs, run the line below to get access to
% CasADi
% addpath /courses/tsfs12/casadi  

%%
filename = 'mprims.mat';
if exist(filename, 'file')
    mp = MotionPrimitives(filename);
else
    
    % Define the initial states and desired goal states for the motion
    % primitives
    theta_init = [0 pi/4 pi/2 3*pi/4 pi -3*pi/4 -pi/2 -pi/4];
    x_vec = [3 2 3 3 3 1 3 3 3 2 3];
    y_vec = [2 2 2 1 1 0 -1 -1 -2 -2 -2];
    th_vec = [0 pi/4 pi/2 0 pi/4 0 -pi/4 0 -pi/2 -pi/4 0];
    lattice = [x_vec;y_vec; th_vec];

    % Vehicle parameters and constraints
    L = 1.5;        % Wheel base (m)
    v = 15;         % Constant velocity (m/s)
    u_max = pi/4;   % Maximum steering angle (rad)

    % Construct a MotionPrimitives object and generate the 
    % motion primitives using the constructed lattice and 
    % specification of the motion primitives
    mp = MotionPrimitives();
    mp.generate_primitives(theta_init, lattice, L, v, u_max);
    % Save the motion primitives to avoid unnecessary re-computation
    mp.save(filename);
end

%% Plot the computed primitives

figure(10)
clf()
mp.plot();
grid on;
xlabel('x');
ylabel('y');
axis('square');
box off
title('Motion Primitives');

%% Create world with obstacles using the BoxWorld class

xx = -2:1:12;
yy = -2:1:12;
th = [0 pi/4 pi/2 3*pi/4 pi -3*pi/4 -pi/2 -pi/4];
lattice = {xx, yy, th};

world = BoxWorld(lattice);

world.add_box(2, 2, 6, 6)
world.add_box(1, 6, 4, 3)
world.add_box(4, 1, 5, 4)

figure(10)
clf()
world.draw()
axis([world.xmin, world.xmax, world.ymin, world.ymax])
xlabel('x');
ylabel('y');

%%
start = [0; 0; 0]; % Initial state
goal = [10; 10; pi/2]; % Final state

n = world.num_nodes();
eps = 1e-5;

% Define the initial and goal state for the graph search by finding the
% node number (column number in world.st_sp) in the world state space
mission.start.id = find(all(abs(world.st_sp - start) < eps, 1));
mission.goal.id = find(all(abs(world.st_sp - goal) < eps, 1));

% Define the function providing possible new states starting from x
f = @(x) next_state(x, world, mp);

% Define heuristics for cost-to-go
cost_to_go = @(x, xg) norm(world.st_sp(1:2, x) - world.st_sp(1:2, xg));

% Solve problem using graph-search strategies from Hand-in Exercise 1

fprintf('Planning ...');
% Start planning
plan = {};
plan{end+1} = BreadthFirst(n, mission, f, [], 2);
plan{end+1} = DepthFirst(n, mission, f, [], 2);
plan{end+1} = Dijkstra(n, mission, f, [], 2);
plan{end+1} = Astar(n, mission, f, cost_to_go, 2);
plan{end+1} = BestFirst(n, mission, f, cost_to_go, 2);
fprintf('Done!\n');

opt_length = plan{3}.length; % Dijkstra is optimal

%% YOUR CODE HERE


