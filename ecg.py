from math import pi, sqrt, atan2, exp, sin
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

class Parameter():
    def __init__(self, time, theta, a, b) -> None:
        self.__time = time
        self.__theta = theta
        self.__a = a
        self.__b = b

    @property
    def time(self):
        return self.__time
    
    @property
    def theta(self):
        return self.__theta
    
    @property
    def a(self):
        return self.__a
    
    @property
    def b(self):
        return self.__b


P = Parameter(time=-0.2, theta=-1/3*pi,  a=1.2, b=0.25)
Q = Parameter(time=0.05, theta=-1/12*pi, a=-5.0, b=0.1)
R = Parameter(time=0.0,  theta=0.0,      a=30.0, b=0.1)
S = Parameter(time=0.05, theta=1/12*pi,  a=-7.5, b=0.1)
T = Parameter(time=0.3,  theta=1/2*pi,   a=0.75, b=0.4)


alpha = lambda x, y: 1 - sqrt(x**2 + y**2)
omega = 1.0
theta = lambda x, y: atan2(y, x)
delta_theta_i = lambda x, y, theta_i: (theta(x, y) - theta_i) % (2 * pi)
f2 = 0.25
z0 = lambda t: 0.15*sin(2*pi*f2*t)

t = np.linspace(0, 5, 1000)

def dxdt(t, x, y):
    alpha(x, y)*x + omega*y

def dydt(t, x, y):
    omega*x + alpha(x, y)*y

def dzdt(z, t, x, y):
    -((P.a * delta_theta_i(x, y, P.theta) * exp(-delta_theta_i(x, y, P.theta)/2*P.b**2) - (z - z0(t))) +
      (Q.a * delta_theta_i(x, y, Q.theta) * exp(-delta_theta_i(x, y, Q.theta)/2*Q.b**2) - (z - z0(t))) +
      (R.a * delta_theta_i(x, y, R.theta) * exp(-delta_theta_i(x, y, R.theta)/2*R.b**2) - (z - z0(t))) +
      (S.a * delta_theta_i(x, y, S.theta) * exp(-delta_theta_i(x, y, S.theta)/2*S.b**2) - (z - z0(t))) +
      (T.a * delta_theta_i(x, y, T.theta) * exp(-delta_theta_i(x, y, T.theta)/2*T.b**2) - (z - z0(t))))
