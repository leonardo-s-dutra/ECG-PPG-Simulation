import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from utils import *


# time vector from 0 to 1
time1 = np.arange(0, 1, 0.008)
period1 = 1
angular_frequency1 = (2*np.pi)/period1

# time vector from -π to +π
time2 = np.arange(-np.pi, np.pi, 0.008)
period2 = 2*np.pi
angular_frequency2 = (2*np.pi)/period2

# auxiliar functions
theta = lambda x, y: np.arctan2(y, x)
delta_theta_i = lambda x, y, theta_i: (theta(x, y) - theta_i)


def calculate_signal(signal: GaussianSignal, time: np.array, angular_frequency: float):
    x = np.cos(angular_frequency*(time - time[0]) - np.pi)
    y = np.sin(angular_frequency*(time - time[0]) - np.pi)
    z = -sum(param.a * delta_theta_i(x, y, param.theta) * np.exp(-delta_theta_i(x, y, param.theta)**2/(2*param.b**2)) for param in signal.gaussians)

    return z


def find_signal_peaks(signal: np.array, time: np.array):
    p_peaks, _ = sp.signal.find_peaks(signal)
    n_peaks, _ = sp.signal.find_peaks(-signal)

    p_coord = list(zip(time[p_peaks], signal[p_peaks]))
    n_coord = list(zip(time[n_peaks], signal[n_peaks]))

    return p_coord, n_coord


def get_signal_parameters(signal_constants: GaussianSignal, signal: np.array) -> GaussianSignal:
    p_coord, n_coord = find_signal_peaks(signal, time2)

    new_signal = calculate_signal(signal_constants, time1, angular_frequency1)
    new_p_coord, new_n_coord = find_signal_peaks(new_signal, time1)
    new_coord = new_p_coord + new_n_coord
    new_coord.sort()

    B = [2*(new_coord[i+1][0] - new_coord[i][0]) for i in range(0, len(new_coord)-1, 2)]
    A = [coord[1] for coord in p_coord]
    THETA = [coord[0] for coord in p_coord]

    parameters = [Gaussian(theta=THETA[i], a=A[i], b=B[i]) for i in range(len(new_p_coord))]

    return GaussianSignal(parameters)


synthethic_ecg = GaussianSignal([
    Gaussian(  theta=-1/3*np.pi,   a=1.2,  b=0.25  ),
    Gaussian(  theta=-1/12*np.pi,  a=-5.0, b=0.1   ),
    Gaussian(  theta=0.0,          a=30.0, b=0.1   ),
    Gaussian(  theta=1/12*np.pi,   a=-7.5, b=0.1   ),
    Gaussian(  theta=1/2*np.pi,    a=0.75, b=0.4   )
])

z = calculate_signal(synthethic_ecg, time2, angular_frequency2)

reconstructed_ecg = get_signal_parameters(synthethic_ecg, z)

z2 = calculate_signal(reconstructed_ecg, time2, angular_frequency2)

#for i in range(5):
#    print('{:.3f}\t-\t{:.3f}'.format(synthethic_ecg.gaussians[i].a, reconstructed_ecg.gaussians[i].a))
#print('\n')
#
#for i in range(5):
#    print('{:.3f}\t-\t{:.3f}'.format(synthethic_ecg.gaussians[i].theta, reconstructed_ecg.gaussians[i].theta))
#print('\n')
#
#for i in range(5):
#    print('{:.3f}\t-\t{:.3f}'.format(synthethic_ecg.gaussians[i].b, reconstructed_ecg.gaussians[i].b))
#print('\n')

plt.plot(time2, z)
plt.plot(time2, z2)
#plt.plot(time2[p_peaks], z[p_peaks], 'x')
#plt.plot(time2[n_peaks], z[n_peaks], 'x')

#ax = plt.axes(projection='3d')
#ax.plot3D(x, y, z, 'gray')
plt.show()
