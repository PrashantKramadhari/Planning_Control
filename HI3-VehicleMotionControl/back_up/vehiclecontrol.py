"""Support files for implementing and simulating controllers
for simple vehicles"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# from scipy.optimize import brenth
# import warnings
import time


class Timer:
    """Simple timer class with tic/toc functionality"""
    t0 = 0.0
    dt = 0.0

    def tic(self):
        """Start timing"""
        self.t0 = time.time()

    def toc(self):
        """Return time since last toc"""
        self.dt = time.time() - self.t0
        return self.dt


class EulerForward:
    """Simple Euler forward integrator class"""
    t = 0
    y = 0
    Ts = -1

    def __init__(self, f):
        self.f = f

    def set_Ts(self, Ts):
        self.Ts = Ts
        return self

    def set_initial_value(self, y, t):
        self.t = t
        self.y = np.array(y)

    def integrate(self, Tend):
        while self.t < Tend:
            if self.t + self.Ts < Tend:
                dt = self.Ts
            else:
                dt = Tend - self.t

            self.y = self.y + self.f(self.t, self.y) * dt
            self.t = self.t + dt

    def successful(self):
        return True


class RungeKutta:
    """Simple Runge-Kutta integrator class"""
    t = 0
    y = 0
    Ts = -1

    def __init__(self, f):
        self.f = f

    def set_Ts(self, Ts):
        self.Ts = Ts
        return self

    def set_initial_value(self, y, t):
        self.t = t
        self.y = y

    def integrate(self, Tend):
        while self.t < Tend:
            if self.t + self.Ts < Tend:
                dt = self.Ts
            else:
                dt = Tend - self.t

            k1 = self.f(self.t, self.y)
            k2 = self.f(self.t, self.y + dt / 2 * k1)
            k3 = self.f(self.t, self.y + dt / 2 * k2)
            k4 = self.f(self.t, self.y + dt / 3 * k3)

            self.y = self.y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.t = self.t + dt

    def successful(self):
        return True


class VehicleModelBase(ABC):
    """Abstract base class for vehicle models

    Derived classes must implement the method
        dw/dt = dx(w, u)
    where w is the state and u the control inputs. Defined in the class as
        def dx(self, w, u)
    and returns a numpy array with the time derivatives of the states.
    """

    _u = None
    _controller = None

    def __init__(self, Ts):
        # self.integrator = ode(lambda t, w: self.dx(t, w, self._u)).set_integrator('vode', method='bdf')
        # self.integrator = EulerForward(lambda t, w: self.dx(t, w, self._u)).set_Ts(Ts)
        self.integrator = RungeKutta(lambda t, w: self.dx(t, w, self._u)).set_Ts(Ts)

    @property
    def Ts(self):
        return self.integrator.Ts

    @Ts.setter
    def Ts(self, Ts):
        self.integrator.set_Ts(Ts)

    def set_attributes(self, opts):
        for ki in opts:
            setattr(self, ki, opts[ki])
        return self

    def set_state(self, w0, t0=0.0):
        self.integrator.set_initial_value(w0, t0)

    @abstractmethod
    def dx(self, t, w, u):
        pass

    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, controller):
        if not isinstance(controller, ControllerBase):
            raise RuntimeError(
                'Controller has to be derived from class ControllerBase')
        self._controller = controller

    def simulate_step(self, Tend):
        self.integrator.integrate(Tend)
        return self.integrator.successful()

    def get_state(self):
        """Return current time and state for the vehicle."""
        return self.integrator.t, self.integrator.y

    @property
    def t(self):
        """Returns current time."""
        return self.integrator.t

    @property
    def w(self):
        """Returns vehicle state."""
        return self.integrator.y

    def simulate(self, w0, T, dt, t0=0.0):
        """Simulate vehicle and controller.

        Simulates the vehicle and controller until specified stop time
        or when the controller stops, i.e., when the controller method run
        returns false.

        Inputs:
          w0 - Initial state
          T  - Stop time
          dt - Controller sampling period
          t0 - Start time (default: 0.0)

        Outputs:
          t - Simulation time vector (sampled by dt)
          w - state trajectories
          u - control signals
        """
        if not isinstance(self._controller, ControllerBase):
            raise RuntimeError('Controller has to be set before simulating, call set_controller first.')

        self._controller.u_time = 0.0

        self.set_state(w0, t0)
        sim = []
        t = []
        u = []
        successful = True
        while ((successful and self.t < T) and self._controller.run(self.t, self.w)):
            self._u = self._controller.u_timed(self.t, self.w)
            u.append(self._u)
            sim.append(self.w)
            t.append(self.t)
            successful = self.simulate_step(self.t + dt)

        u.append(self._controller.u(self.t, self.w))
        sim.append(self.w)
        t.append(self.t)

        return np.array(t), np.array(sim), np.array(u)


class SingleTrackModel(VehicleModelBase):
    L = 2
    amax = np.inf
    amin = -np.inf
    steer_limit = np.pi / 3

    def __init__(self, Ts=0.1):
        """ Create Single track kinematic car

             obj = SingleTrackModel(Ts)
             Ts - controller sample time"""
        super(SingleTrackModel, self).__init__(Ts)

    def dx(self, t, w, u):
        """Dynamic equation for single-track model

           dw = obj.dx(w, u)

           Input:
             w  - state (x, y, theta, v)
             u - control input (delta, acceleration)

           Output:
             dw - dw/dt"""
        x, y, theta, v = w
        delta = np.max((np.min((u[0], self.steer_limit)), -self.steer_limit))
        a = np.max((np.min((u[1], self.amax)), self.amin))

        dx = np.array([v * np.cos(theta),
                       v * np.sin(theta),
                       v / self.L * np.tan(delta),
                       a])
        return dx


class ControllerBase(ABC):
    def __init__(self):
        self.u_time = 0.0
        self.u_timer = Timer()

    def u_timed(self, t, w):
        self.u_timer.tic()
        u = self.u(t, w)
        self.u_time = self.u_time + self.u_timer.toc()
        return u

    @abstractmethod
    def u(self, t, w):
        pass

    def run(self, _t, _w):
        return True


class PurePursuitControllerBase(ControllerBase):
    def __init__(self, pursuit_point_fig=None):
        super().__init__()
        self._pp_fig = pursuit_point_fig

        if self._pp_fig:
            # Create placeholder plot objects for car, pursuit-point, horizon circle, and travelled path
            self._car_point = plt.plot([], [], "ro")
            self._pursuit_point = plt.plot([], [], "bo")
            self._pursuit_horizon = plt.plot([], [], "k", lw=0.5, ls=(0, (10, 10)))
            self._car_path = plt.plot([], [], "k", lw=0.75)

    def _pursuit_plot(self, p_car, p_purepursuit):
        "Helper function to illustrate pursuit-point selection"

        if self._pp_fig:  # Is pursuit-point illustration activated?
            self._car_point[0].set_data([p_car[0]], [p_car[1]])  # Move car
            self._pursuit_point[0].set_data(
                [p_purepursuit[0]], [p_purepursuit[1]]
            )  # Move pursuit-point
            phi = np.linspace(0, 2 * np.pi, 40)
            self._pursuit_horizon[0].set_data(
                np.cos(phi) * self.l + p_car[0], np.sin(phi) * self.l + p_car[1]
            )  # Horizon circle
            dd = self._car_path[0].get_data()
            self._car_path[0].set_data(
                (np.hstack((dd[0], [p_car[0]])), np.hstack((dd[1], [p_car[1]])))
            )  # Travelled path

            # Update plot
            self._pp_fig.canvas.draw()
            self._pp_fig.canvas.flush_events()

class StateFeedbackControllerBase(ControllerBase):
    def __init__(self, pursuit_point_fig=None):
        super().__init__()
        self._pp_fig = pursuit_point_fig

        if self._pp_fig:
            # Create placeholder plot objects for car, pursuit-point, horizon circle, and travelled path
            self._car_point = plt.plot([], [], "ro")
            self._car_path = plt.plot([], [], "k", lw=0.75)

    def _plot(self, p_car):
        "Helper function to illustrate pursuit-point selection"

        if self._pp_fig:  # 
            self._car_point[0].set_data([p_car[0]], [p_car[1]])  # Move car
            dd = self._car_path[0].get_data()
            self._car_path[0].set_data(
                (np.hstack((dd[0], [p_car[0]])), np.hstack((dd[1], [p_car[1]])))
            )  # Travelled path

            # Update plot
            self._pp_fig.canvas.draw()
            self._pp_fig.canvas.flush_events()


def plot_car(w, W, L, *args):
    x, y, theta, _ = w
    p = np.array([x, y])
    heading = np.array([np.cos(theta), np.sin(theta)])
    nc = np.array([-heading[1], heading[0]])

    p_car = np.array([p - W / 2 * nc, p - W / 2 * nc + L * heading,
                      p + W / 2 * nc + L * heading, p + W / 2 * nc, p - W / 2 * nc])
    plt.plot(p_car[:, 0], p_car[:, 1], *args)
