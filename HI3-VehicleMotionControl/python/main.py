#!/usr/bin/env python
# coding: utf-8

# TSFS12 Hand-in exercise 3: Path following for autonomous vehicles

import numpy as np
import matplotlib.pyplot as plt
from vehiclecontrol import ControllerBase, SingleTrackModel, PurePursuitControllerBase, StateFeedbackControllerBase
from splinepath import SplinePath
from scipy.linalg import solve_discrete_are
import math
from seaborn import despine
from scipy import linalg as la
from main_lqr_continuos import LQR
from main_pp import PurePursuitController

# Run if you want plots in external windows
# %matplotlib
plt.ion()

# Run the ipython magic below to activate automated import of modules. Useful if you write code in external .py files.
# %load_ext autoreload
# %autoreload 2


# Make a simple controller and simulate vehicle


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
w0 = [0, 0, 0, 2]
z0 = car.simulate(w0, T=40, dt=0.1, t0=0.0)
t, w, u = z0
M = 10
p = w[::M, 0:2]
nom_path = SplinePath(p)


#Below is for PP
car = SingleTrackModel()
pp_controller = PurePursuitController(l=4, L=car.L, path=nom_path,goal_tol=0.25)
car.controller = pp_controller
w0 = [0, 1, np.pi / 2 * 0.9, 2]  # Sample starting state
z_pp = car.simulate(w0, T=80, dt=0.1, t0=0.0)
print(f"total time in controller : { car.controller.u_time*1000:0.2f} mesk")
t, w, u = z_pp

phi = np.linspace(0, 2 * np.pi, 20)
_, ax = plt.subplots(num=70, clear=True)
ax.plot(p[:, 0], p[:, 1], 'b', lw=0.5)
ax.plot(w[:, 0], w[:, 1], 'k')
ax.plot(w0[0] + car.controller.l *np.cos(phi),w0[1] + car.controller.l *np.cos(phi))
despine()

d = nom_path.path_error(w[:, 0:2])
_, ax = plt.subplots(num=71, clear=True)
ax.plot(t, d[:t.shape[0]]
)
ax.set_xlabel("t [s]")
ax.set_ylabel("error [m]")
#ax.set_yticks([0 , 0.5,1,1.5,2,2.5,3])
ax.set_title("Path error (orthongonal")
despine()


_, ax = plt.subplots(num=72, clear=True)
ax.plot(t, u[:,0] * 180 /np.pi)
ax.set_xlabel("t [s]")
ax.set_ylabel("[]deg]")
ax.set_title("Steer")
despine()

# Below is for LQR
w0 = [0, 1, np.pi / 2 * 0.9, 2]
kd= 1
s = np.linspace(0, nom_path.length, 200)
fig, ax = plt.subplots(num=99, clear=True)
ax.plot(nom_path.x(s), nom_path.y(s), "b", lw=0.5)
ax.plot(nom_path.path[:, 0], nom_path.path[:, 1], "rx", markersize=3)
plt.show()

lq = LQR(
    K=kd, L=car.L, path=nom_path, goal_tol=0.25, pursuit_point_fig=fig)

car = SingleTrackModel().set_attributes({"steer_limit": np.pi/4})
car.controller = lq
car.controller.Ts = 0.1

z_lq = car.simulate(w0, T=80, dt=car.controller.Ts, t0=0.0)
t, w, u = z_lq
path_error = nom_path.path_error(w[:,0:2])

s = np.linspace(0, lq.plan.length, 200)

_, ax = plt.subplots(num=80, clear=True)
ax.plot(lq.plan.x(s), lq.plan.y(s), 'b', lw=0.5)
ax.plot(w[:, 0], w[:, 1], 'k')
ax.set_title("Path length:{:.1f} m ".format(lq.plan.length))
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
despine()


_, ax = plt.subplots(num=81, clear=True)
ax.plot(t[:d.shape[0]], path_error[:d.shape[0]], " b")
ax.plot(t[:d.shape[0]], d[:t.shape[0]], " r")
ax.set_xlabel("t [s]")
ax.set_ylabel("deg]")
ax.set_title("path error")
despine() 

print('BP')