classdef PurePursuitController < ControllerBase
    properties
        l;
        L;
        plan;
        goal_tol;
        s;
    end
    methods
        function obj=PurePursuitController(l, L, path, goal_tol)
            obj = obj@ControllerBase();
            if nargin < 4
                goal_tol = 1;
            end
            obj.l = l;
            obj.L = L;
            obj.plan = path;
            obj.goal_tol = goal_tol;
            obj.s = 0;
        end

        function p_purepursuit = pursuit_point(obj, p_car)
            % p_car - position of vehicle

            s = obj.s; % Last stored path parameter
            path_points = obj.plan.path;  % Points 
            l = obj.l;  % Pure-pursuit look-ahead
            % Your code here
            p_purepursuit = [0, 0]; 
        end
     
        function delta = pure_pursuit_control(obj, dp, theta)
            % dp - vector p_purepursuit - p_car
            % theta - heading of vehicle

            % Your code here to compute new steering angle
            delta = 0;
        end

        function c = u(obj, t, w)
            p_car = w(1:2);
            theta = w(3);      

            % Your code here to compute steering angle, use the functions
            % obj.pursuit_point() and obj.pure_pursuit_control() you 
            % have written above.

            delta = 0;
            acc = 0;
            c = [delta, acc];
        end
    
        function r = run(obj, t, w)
            % Function that returns true until goal is reached
            p_car = w(1:2);
            p_goal = obj.plan.path(end, :);
            r = norm(p_car - p_goal) > obj.goal_tol;
        end
    end
end
