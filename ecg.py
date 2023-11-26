import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from utils import *
from dataset import Get_Real_Dataset


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
    z = sum(param.a * np.exp(-delta_theta_i(x, y, param.theta)**2/(2*param.b**2)) for param in signal.gaussians)
    return z


def find_signal_peaks(signal: np.array, time: np.array):
    p_peaks, _ = sp.signal.find_peaks(signal, prominence=0.1)
    n_peaks, _ = sp.signal.find_peaks(-signal, prominence=0.1)

    p_coord = list(zip(time[p_peaks], signal[p_peaks]))
    n_coord = list(zip(time[n_peaks], signal[n_peaks]))

    return p_coord, n_coord

def find_last_zero(time: np.array, signal: np.array):
    for i in range(0, len(signal)):
        if signal[i] > 0.34:
            return time[i]


def get_signal_parameters(signal: np.array):
    p_coord, n_coord = find_signal_peaks(signal, time1)
    coord = p_coord + n_coord
    coord.sort()

    B = []
    B.append(2*(coord[0][0] - find_last_zero(time1, signal)))
    B.append(2*(coord[2][0] - coord[1][0]))
    B.append(2*(coord[3][0] - coord[2][0]))
    B.append(2*(coord[3][0] - coord[2][0]))
    B.append(2*(coord[4][0] - coord[3][0]))

    A = [coord_[1] for coord_ in coord]
    THETA = [coord_[0]*(2*np.pi)-np.pi for coord_ in coord]

    parameters = [Gaussian(theta=THETA[i], a=A[i], b=B[i]) for i in range(len(coord))]

    return GaussianSignal(parameters)


synthethic_ecg = GaussianSignal([
    Gaussian(  theta=-1/3*np.pi,   a=0.2,   b=0.25  ),
    Gaussian(  theta=-1/12*np.pi,  a=-0.4,  b=0.1   ),
    Gaussian(  theta=0.0,          a=1.2,   b=0.1   ),
    Gaussian(  theta=1/12*np.pi,   a=-0.5,  b=0.1   ),
    Gaussian(  theta=1/2*np.pi,    a=0.3,   b=0.4   )
])

_, real_ecg = Get_Real_Dataset()
real_ecg = real_ecg - 0.3

real_ecg_parameters = get_signal_parameters(real_ecg)

reconstructed_ecg = calculate_signal(real_ecg_parameters, time1, angular_frequency1)

plt.plot(time1, real_ecg)
plt.plot(time1, reconstructed_ecg)
plt.show()
