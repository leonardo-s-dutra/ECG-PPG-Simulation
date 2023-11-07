import matplotlib.pyplot as plt
import numpy as np
import math

time:list = np.arange(0,1,0.008)   # start,stop,step

T=1
angular_frequency = (2*np.pi)/T

omega = list()
x = list()
y = list()
z = list()

_a = [1, 0.1999]
_b = [0.6303, 1.0225]
_o = [-1.5161, 0.8186]

for t in time:
    x_t = np.cos(angular_frequency*(t - time[0]) - np.pi)
    y_t = np.sin(angular_frequency*(t - time[0]) - np.pi)
    o_t = np.arctan2(y_t, x_t)

    z_1 = _a[0]*math.exp(-((o_t - _o[0])**2)/(2*_b[0]**2))
    z_2 = _a[1]*math.exp(-((o_t - _o[1])**2)/(2*_b[1]**2))

    x.append(x_t)
    y.append(y_t)
    z.append(z_1 + z_2)
    omega.append(o_t)

ax = plt.axes(projection='3d')
ax.plot3D(x, y, z, 'gray')
#plt.plot(time, z)
#plt.plot(time, omega)
#plt.plot(time, x)
#plt.plot(time, y)
plt.show()