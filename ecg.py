from math import pi, sqrt, atan2

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
    Parameter(time=-0.2, theta=-1/3*pi,  a=1.2, b=0.25), # P
    Parameter(time=0.05, theta=-1/12*pi, a=-5.0, b=0.1), # Q
    Parameter(time=0.0,  theta=0.0,      a=30.0, b=0.1), # R
    Parameter(time=0.05, theta=1/12*pi,  a=-7.5, b=0.1), # S
    Parameter(time=0.3,  theta=1/2*pi,   a=0.75, b=0.4), # T
]

alpha = lambda x, y: 1 - sqrt(x**2 + y**2)
omega = 1.0
theta = lambda x, y: atan2(y, x)
delta_theta_i = lambda x, y, param: (theta(x, y) - param.theta) % (2 * pi)