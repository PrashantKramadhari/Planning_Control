% Path regression and class prediction using Gaussian processes and the
% data from the I-80 data set from the U.S. Department of 
% Transportation. The data can be downloaded from the course directory in 
% Lisam, and are available in the directory /courses/tsfs12/i80_data in 
% the student labs at campus. 
% I-80 data set citation: U.S. Department of Transportation Federal Highway
% Administration. (2016). Next Generation Simulation (NGSIM) Vehicle
% Trajectories and Supporting Data. [Dataset]. Provided by ITS DataHub
% through Data.transportation.gov. Accessed 2020-09-29 from
% http://doi.org/10.21949/1504477. More details about the data set are 
% available through this link. 
% 
% A simplified version of the method presented in Tiger, M., & F. Heintz: 
% ''Online sparse Gaussian process regression for trajectory modeling''. 
% International Conference on Information Fusion (FUSION), pp. 782-791,
% 2015, is used in the exercise.

clear;
close all;

i80_data_dir = 'i80_data/';
% Uncomment if in the student labs at campus:
% dir = '/courses/tsfs12/i80_data/';

% Load driver paths from I-80 data set
[tracks_s_I80,tracks_x_I80,tracks_y_I80,lane_id_I80,N_paths] = ...
    load_I80_gp_dataset(i80_data_dir);

% Plot the paths in the loaded data set

figure;
hold on;
colors = {'b','m','r','m','g','y','k'};
for i = 1:N_paths
    if lane_id_I80{i}(1) == 1
        plot(tracks_x_I80{i}, ...
            tracks_y_I80{i},colors{1},'linewidth',1)
    elseif lane_id_I80{i}(1) == 2
        plot(tracks_x_I80{i}, ...
            tracks_y_I80{i},colors{2},'linewidth',1)
    elseif lane_id_I80{i}(1) == 3
        plot(tracks_x_I80{i}, ...
            tracks_y_I80{i},colors{3},'linewidth',1)
    elseif lane_id_I80{i}(1) == 4
        plot(tracks_x_I80{i}, ...
            tracks_y_I80{i},colors{4},'linewidth',1)
    elseif lane_id_I80{i}(1) == 5
        plot(tracks_x_I80{i}, ...
            tracks_y_I80{i},colors{5},'linewidth',1)
    elseif lane_id_I80{i}(1) == 6
        plot(tracks_x_I80{i}, ...
            tracks_y_I80{i},colors{6},'linewidth',1)
    elseif lane_id_I80{i}(1) == 7
        plot(tracks_x_I80{i}, ...
            tracks_y_I80{i},colors{7},'linewidth',1)
    end
end

% Plot I80 lane boundaries
lb = [-0.5385917537746898
    3.231216608594652
    7.001024970963993
    10.770833333333336
    14.540641695702677
    18.310450058072018
    22.08025842044136];

y_lane = linspace(-100,600,2);

for i = 1:length(lb)
    plot(lb(i)*ones(2,1),y_lane,'k--','linewidth',2);
end
xlabel('x [m]')
ylabel('y [m]')
grid on;

%% Extract paths corresponding to specific lane-change scenario

init_lane =  7;     % Initial lane, the on-ramp is 7
final_lane = 6;     % Final lane (1-6), counted from left to right

[tracks_s,tracks_x,tracks_y,lane_id,N_paths_gp] = ...
    load_I80_gp_dataset_lanes(tracks_s_I80,tracks_x_I80,tracks_y_I80, ...
    lane_id_I80,init_lane,final_lane);

if N_paths_gp == 0
    disp('No paths with specified lane change exist.');
end

s_train = [];
x_train = [];
y_train = [];

N_samples_gp = 8; % Number of samples along each driver path for GP regression

for i = 1:N_paths_gp
    s0 = [0; rand(N_samples_gp-2,1); 1];
    s_train = [s_train; s0];
    x_train = [x_train; interp1(tracks_s{i},tracks_x{i},s0)];
    y_train = [y_train; interp1(tracks_s{i},tracks_y{i},s0)];
end


%% GP model

Ns = 100; % Number of predictive samples in the interval [0,1]
s_samp = linspace(0,1,Ns)';

% Compute GP model for the x-coordinate along the path

disp('Perform GP hyperparameter optimization for x path...')
hyp_param_x = gp_reg_hp_fit(s_train,x_train);
disp('Finished')

% Compute score for fit
gp_score_x = gp_score(gp_reg_pred(s_train,x_train,s_train,hyp_param_x),x_train);

disp('Perform GP prediction for x path...')
[path_x_mean, path_x_var] = gp_reg_pred(s_train,x_train,s_samp,hyp_param_x);
disp('Finished')

% Compute GP model for the y-coordinate along the path

disp('Perform GP hyperparameter optimization for y path...')
hyp_param_y = gp_reg_hp_fit(s_train,y_train);
disp('Finished')

% Compute score for fit
gp_score_y = gp_score(gp_reg_pred(s_train,y_train,s_train,hyp_param_y),y_train);

disp('Perform GP prediction for y path...')
[path_y_mean, path_y_var] = gp_reg_pred(s_train,y_train,s_samp,hyp_param_y);
disp('Finished')


%% Plot the training data points and the predicted values from the GP model

% YOUR CODE HERE


%% Compute prediction of lane-change class

disp('Compute fit of the paths to the GP models for other lane changes...')
pred_score = predict_scenario_score(s_train,x_train,y_train,...
    hyp_param_x,hyp_param_y,tracks_s_I80,tracks_x_I80,tracks_y_I80,...
    lane_id_I80);
disp('Finished')

pred_score_softmax = softmax(pred_score);
[max_score,max_idx] = max(pred_score_softmax);

% Plot the softmax outputs to investigate which scenario that is most
% likely
figure;
bar(1:6, pred_score_softmax);
hold on
plot(max_idx,max_score,'r+','linewidth',2)
grid on
xlabel('Lane change')
ylabel('Score')


%% Auxiliary functions used for the GP model and path investigations

% Function to compute softmax values for vector z
function r = softmax(z)

exp_z = exp(z);
r = exp_z/sum(exp_z);

end

% Compute the score of the fit between predicted and true outputs
function score = gp_score(y_pred,y_true)

score = 1-sum((y_true-y_pred).^2)/sum((y_true-mean(y_true)).^2);

end

% Compute the fit of the paths for the different lane changes
function pred_score = predict_scenario_score(s_train,x_train,y_train,...
    hyp_param_x,hyp_param_y,tracks_s_I80,tracks_x_I80,tracks_y_I80,...
    lane_id_I80)

pred_score = [];
    
for i=1:6
    score = [];
    
    [tracks_s,tracks_x,tracks_y,~,N_paths_gp] = ...
    load_I80_gp_dataset_lanes(tracks_s_I80,tracks_x_I80,tracks_y_I80,...
    lane_id_I80,7,i);

    for j=1:5:N_paths_gp
        score = [score; gp_score(gp_reg_pred(s_train,x_train,tracks_s{j},...
            hyp_param_x),tracks_x{j});
            gp_score(gp_reg_pred(s_train,y_train,tracks_s{j},hyp_param_y),...
            tracks_y{j})];
    end
    pred_score = [pred_score; mean(score)];
end

end


% Load the complete I80 data set
function [tracks_s_I80,tracks_x_I80,tracks_y_I80,lane_id_I80,N_paths]...
    = load_I80_gp_dataset(i80_data_dir)

if verLessThan('matlab','9.6')
    I80_data = csvread([i80_data_dir '0500pm-0515pm/trajectories-0500-0515.csv'],1);
else
    I80_data = readmatrix([i80_data_dir '0500pm-0515pm/trajectories-0500-0515.csv']);
end

veh_idx = find(diff(I80_data(:,1)) ~= 0);
veh_idx = [0; veh_idx; length(I80_data(:,1))];
for i=1:length(veh_idx)-1
   tracks_x_I80{i} = I80_data(veh_idx(i)+1:veh_idx(i+1),5)/3.28; % Unit conversion
   tracks_y_I80{i} = I80_data(veh_idx(i)+1:veh_idx(i+1),6)/3.28; % Unit conversion
   lane_id_I80{i} = I80_data(veh_idx(i)+1:veh_idx(i+1),14);
end

N_paths = length(veh_idx)-1;

for i = 1:length(tracks_x_I80)
    path_length = cumsum(sqrt(sum([diff(tracks_x_I80{i}) ...
        diff(tracks_y_I80{i})].^2,2)));
    
    % Normalize path coordinate to [0,1]
    tracks_s_I80{i} = [0; path_length/path_length(end)];
end

end

% Extract part of I80 data set, with specified initial and final lanes
function [tracks_s,tracks_x,tracks_y,lane_id,N_paths_gp] =...
    load_I80_gp_dataset_lanes(tracks_s_I80,tracks_x_I80,tracks_y_I80,...
    lane_id_I80,init_lane,final_lane)

N_paths = length(tracks_x_I80);

tracks_s = [];
tracks_x = [];
tracks_y = [];
lane_id = [];

k = 1;
for i = 1:N_paths
    if lane_id_I80{i}(1) == init_lane && lane_id_I80{i}(end) == final_lane
        if any(diff(tracks_s_I80{i}) < 1e-4)
            continue;
        end
        
        tracks_s{k} = tracks_s_I80{i};
        tracks_x{k} = tracks_x_I80{i};
        tracks_y{k} = tracks_y_I80{i};
        lane_id{k} = lane_id_I80{i};
        k = k+1;
    end
end

N_paths_gp = length(tracks_x);

end

% Optimize the hyperparameters
function hyp_param = gp_reg_hp_fit(x,y)

mean_fcn = [];
cov_fcn = @covSEiso;
lik_fcn = @likGauss;

hyp_init = struct('mean', [], 'cov', [0 0.1], 'lik', -1);

% Use GPML Toolbox for optimization of hyperparameters
hyp_opt = minimize(hyp_init, @gp, -200, @infGaussLik, mean_fcn, cov_fcn, lik_fcn, x, y);

% Save hyperparameters in a struct
hyp_param.sigma_f = exp(hyp_opt.cov(2));
hyp_param.l = exp(hyp_opt.cov(1));
hyp_param.sigma_e = exp(hyp_opt.lik);

end

% Compute predicted outputs (mean and variance) for inputs in xs
function [ys_pred_mean, ys_pred_var] = gp_reg_pred(x,y,xs,hyp_param)

N = length(x);
Ns = length(xs);

sigma_f = hyp_param.sigma_f;
l = hyp_param.l;

k_f = @(x_p,x_q) sigma_f^2*exp(-1/(2*l^2)*(x_p-x_q).^2);

K = bsxfun(k_f, x, x');

sigma_e = hyp_param.sigma_e;

% Algorithm 2.1 in Rasmussen, C. E., & C. K. I. Williams: Gaussian
%  Processes for Machine Learning. MIT Press, 2006.

L = chol(K + sigma_e^2*eye(N),'lower');
alpha = L'\(L\y);

for i = 1:Ns
    k_star = bsxfun(k_f, x, xs(i));
    ys_pred_mean(i,1) = k_star'*alpha;
    v = L\k_star;
    ys_pred_var(i,1) = k_f(xs(i),xs(i))-v'*v+sigma_e^2;
end

end