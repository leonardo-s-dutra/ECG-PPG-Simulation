from math import pi, sqrt, atan2

parameters = [
    {'time': -0.2,  'theta': -1/3*pi,   'a': 1.2,   'b': 0.25   }, # P
    {'time': -0.05, 'theta': -1/12*pi,  'a': -5.0,  'b': 0.1    }, # Q
    {'time': 0.0,   'theta': 0.0,       'a': 30.0,  'b': 0.1    }, # R
    {'time': 0.05,  'theta': 1/12*pi,   'a': -7.5,  'b': 0.1    }, # S
    {'time': 0.3,   'theta': 1/2*pi,    'a': 0.75,  'b': 0.4    }, # T
]

alpha = lambda x, y: 1 - sqrt(x**2 + y**2)
omega = 1.0
theta = lambda x, y: atan2(y, x)
delta_theta_i = lambda i: (theta - parameters[i]['theta']) % (2 * pi)
