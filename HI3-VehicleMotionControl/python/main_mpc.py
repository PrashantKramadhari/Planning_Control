#!/usr/bin/env python
# coding: utf-8

# TSFS12 Hand-in exercise 3, extra assignment: Path following for autonomous vehicles with MPC controller

import numpy as np
import matplotlib.pyplot as plt
from vehiclecontrol import ControllerBase, SingleTrackModel, MPCBase
from splinepath import SplinePath
import casadi
import math
import matplotlib.animation as animation


plt.ion()
# %matplotlib  # Run if you want plots in external windows


# Run the ipython magic below to activate automated import of modules. Useful if you write code in external .py files.
# %load_ext autoreload
# %autoreload 2


# Make a simple controller and simulate vehicle

# Simulate a sample path to follow


class MiniController(ControllerBase):
    def __init__(self):
        super().__init__()

    def u(self, t, w):
        a = 0.0
        if t < 10:
            u = [np.pi / 180 * 10, a]
        elif 10 <= t < 20:
            u = [-np.pi / 180 * 11, a]
        elif 20 <= t < 23:
            u = [-np.pi / 180 * 0, a]
        elif 23 <= t < 40:
            u = [-np.pi / 180 * 15, a]
        else:
            u = [-np.pi / 180 * 0, a]
        return u


opts = {"L": 2, "amax": np.inf, "amin": -np.inf, "steer_limit": np.pi / 3}

car = SingleTrackModel().set_attributes(opts)
car.Ts = 0.1
car.controller = MiniController()
w0 = np.array([0, 0, 0, 2])
z0 = car.simulate(w0, T=40, dt=0.1, t0=0.0)
t, w, u = z0
M = 10
p = w[::M, 0:2]
nom_path = SplinePath(p)


s = np.linspace(0, nom_path.length, 100)

_, ax = plt.subplots(num=10, clear=True)
ax.plot(nom_path.x(s), nom_path.y(s))
ax.plot(p[:, 0], p[:, 1], "rx")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Path from simple controller")
_ = ax.axis("square")

# Make an MPC path following controller

# Make a MPC controller using the template class below. Parameters for the controller are
# * gamma_d - Weight in the loss-function for _distance errors_
# * gamma_theta - Weight in the loss-function for _heading errors_
# * gamma_u - Weight in the loss-function for _control signal_ (steer angle)
# * L - wheel base
# * steer_limit - Steer limits (in radians) for the control signal

# Arguments to the controller is provided in a dictionary like this

opts = {
    "h_p": 10,
    "gamma_d": 1,
    "gamma_theta": 1,
    "gamma_u": 1,
    "L": 2,  # Nominally, use the same as in the car
    "steer_limit": np.pi / 4,  # Nominally, use the same in the car
}


class ModelPredictiveController(MPCBase):
    def __init__(self, controller_params, path=None, goal_tol=1, dt=0.1,car_point_fig=None):
        super().__init__(car_point_fig)

        self.plan = path
        self.gamma_d = controller_params["gamma_d"]
        self.gamma_theta = controller_params["gamma_theta"]
        self.gamma_u = controller_params["gamma_u"]
        self.L = controller_params["L"]
        self.steer_limit = controller_params["steer_limit"]

        self.sample_rate = dt
        self.prediction_horizon = controller_params["h_p"]
        self.N = int(self.prediction_horizon / dt)

        self.goal_tol = goal_tol
        self.d = []
        self.s0 = 0
        self.optimizer = self.construct_problem()

    def heading_error(self, theta, s):
        """Compute theta error
        Inputs
            theta - current heading angle
            s - projection point on path

        Outputs
            theta_e - heading error angle
        """

        # YOUR CODE HERE
        # Use your code from state-feedback controller in the basic exercise

        theta_e = 0.0

        h_s, nc_s = self.plan.heading(s)
        theta_s = np.arctan2(h_s[1], h_s[0])
        
        h = np.array([math.cos(theta), math.sin(theta)])
        
        cos_theta_e = np.dot(h_s, h)
        sin_theta_e = np.cross(h_s, h)
        
        theta_e = math.atan(sin_theta_e / cos_theta_e)
        return theta_e

    def construct_problem(self):
        """Formulate optimal control problem"""

        dt = self.sample_rate

        # Create an casadi.Opti instance.
        opti = casadi.Opti("conic")

        d0 = opti.parameter()
        th0 = opti.parameter()
        v = opti.parameter()
        curvature = opti.parameter(self.N)

        X = opti.variable(2, self.N + 1)
        proj_error = X[0, :]
        head_error = X[1, :]

        # Control variable (steering angle)
        Delta = opti.variable(self.N)

        # Goal function we wish to minimize
        ### YOUR CODE HERE ###
        # Use the casadi.sumsqr function to compute sum-of-squares
        J = self.gamma_d * casadi.sumsqr(proj_error) + self.gamma_theta * casadi.sumsqr(head_error) + \
            self.gamma_u * casadi.sumsqr(Delta)

        opti.minimize(J)

        # Simulate the system forwards using RK4 and the implemented error model.
        for k in range(self.N):
            k1 = self.error_model(X[:, k], v, Delta[k], curvature[k])
            k2 = self.error_model(X[:, k] + dt / 2 * k1, v, Delta[k], curvature[k])
            k3 = self.error_model(X[:, k] + dt / 2 * k2, v, Delta[k], curvature[k])
            k4 = self.error_model(X[:, k] + dt * k3, v, Delta[k], curvature[k])
            x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next)

        # Problem constraints.
        opti.subject_to(proj_error[0] == d0)
        opti.subject_to(head_error[0] == th0)
        opti.subject_to(opti.bounded(-self.steer_limit, Delta, self.steer_limit))

        # The cost function is quadratic and the problem is linear by design,
        # this is utilized when choosing solver.
        # Other possible solvers provided by CasADi include: 'qpoases' and 'ipopt'...

        opts_dict = {
            "print_iter": False,
            "print_time": 0,
            "constr_viol_tol": 1e-12,
            # "max_iter": 100
        }

        opti.solver("qrqp", opts_dict)

        return opti.to_function("f", [d0, th0, v, curvature], [Delta])

    def error_model(self, w, v, delta, curvature):
        """Error model describing how the distance and heading error evolve for a certain input

        Input:
            w = (d, theta_e)
            v - velocity
            delta - input, steering angle
            curvature - current curvature

        Output:
            Casadi vector of Time derivative of d and theta_e
        """

        # YOUR CODE HERE
        s_dot = v

        d_dot = 0
        theta_e_dot = 0

        d_dot = v * w[1]
        #theta_e_dot = v * (delta - curvature * (1 - curvature * w[0]))
        theta_e_dot = v*delta/(self.L) - s_dot*curvature

        #theta_e_dot = v * (delta/self.L - curvature)


        return casadi.vertcat(d_dot, theta_e_dot)

    def u(self, t, w):
        p_car = w[0:2]
        theta = w[2]
        v = w[3]

        # Compute d and theta_e errors as in the basic exercise state-feedback controller
        # YOUR CODE HERE
        s, d = self.plan.project(p_car, self.s0)
        self.s0 = s

        theta_e = self.heading_error(theta, s)
        s_i = 0  # Position for start of prediction

        # Solve optimization problem over the prediction-horizon
        s_horizon = np.linspace(s_i, s_i + self.N * v * self.sample_rate, self.N)
        Delta = self.optimizer(d, theta_e, v, self.plan.c(s_horizon))  # YOUR CODE HERE, call self.optimizer() with proper arguments

        delta = float(Delta[0])


        self._plot(p_car)  # Included for animation purposes.


        # Collect the controller output
        acc = 0
        self.d.append(d)

        return np.array([delta, acc])

    def run(self, t, w):
        p_goal = self.plan.path[-1, :]
        p_car = w[0:2]
        dp = p_car - p_goal
        dist = np.sqrt(dp.dot(dp))
        if dist < self.goal_tol:
            return False
        else:
            return True


# Simulate

opts = {
    "h_p": 10,
    "gamma_d": 2,
    "gamma_theta": 10,
    "gamma_u": 0.01,
    "L": car.L,
    "steer_limit": np.pi / 4,
}

w0 = [0, 1, np.pi / 2 * 0.9, 2]  # (x, y, theta, v)
s = np.linspace(0, nom_path.length, 200)
fig, ax = plt.subplots(num=99, clear=True)
ax.plot(nom_path.x(s), nom_path.y(s), "b", lw=0.5)
ax.plot(nom_path.path[:, 0], nom_path.path[:, 1], "rx", markersize=3)

plt.show()

car = SingleTrackModel().set_attributes({"steer_limit": opts["steer_limit"]})

mpc = ModelPredictiveController(controller_params=opts, path=SplinePath(p), dt=0.1,car_point_fig=fig)
car.controller = mpc

z_mpc = car.simulate(w0, T=80, dt=mpc.sample_rate, t0=0.0)
print(f"Total time in controller: {mpc.u_time:.2f} sek")

plt.show()


