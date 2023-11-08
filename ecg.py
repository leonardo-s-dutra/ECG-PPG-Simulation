import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
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

parameters = [
    Parameter(  time=-0.2,    theta=-1/3*np.pi,   a=1.2,  b=0.25  ),
    Parameter(  time=-0.05,   theta=-1/12*np.pi,  a=-5.0, b=0.1   ),
    Parameter(  time=0.0,     theta=0.0,          a=30.0, b=0.1   ),
    Parameter(  time=0.05,    theta=1/12*np.pi,   a=-7.5, b=0.1   ),
    Parameter(  time=0.3,     theta=1/2*np.pi,    a=0.75, b=0.4   )
]

period = 1.0
angular_frequency = (2*np.pi)/period

theta = lambda x, y: np.arctan2(y, x)
delta_theta_i = lambda x, y, theta_i: (theta(x, y) - theta_i)

time = np.arange(0, 1, 0.008)

x = np.cos(angular_frequency*(time - time[0]) - np.pi)
y = np.sin(angular_frequency*(time - time[0]) - np.pi)
z = -sum(param.a * delta_theta_i(x, y, param.theta) * np.exp(-delta_theta_i(x, y, param.theta)**2/(2*param.b**2)) for param in parameters)

ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'gray')
plt.show()
