import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy as sp
import numpy as np
import math

@dataclass
class PPG:
    peak_1: float
    peak_2: float
    theta_1: float
    theta_2: float
    b_1: float
    b_2: float

    #def __post_init__(self):
    #    self.calculate_b()

    #def calculate_b(self):
    #    m = sp.interpolate.interp1d([-np.pi, np.pi],[0,1])
    #    self.b_1 = 2 * (float(m(self.theta_1)))
    #    self.b_2 = 2 * (1 - float(m(self.theta_2)))


def calculate_ppg(ppg: PPG, time:list, angular_frequency:float):
    z = list()
    for t in time:
        x_t = np.cos(angular_frequency*(t - time[0]) - np.pi)
        y_t = np.sin(angular_frequency*(t - time[0]) - np.pi)
        o_t = np.arctan2(y_t, x_t)

        z_1 = ppg.peak_1*math.exp(-((o_t - ppg.theta_1)**2)/(2*ppg.b_1**2))
        z_2 = ppg.peak_2*math.exp(-((o_t - ppg.theta_2)**2)/(2*ppg.b_2**2))

        z.append(z_1 + z_2)

    return z
    
def find_peaks(z: list, time:list):
    z = np.asarray(z) 

    p_peaks, _ = sp.signal.find_peaks(z)
    n_peaks, _ = sp.signal.find_peaks(-z)

    p_coord = list(zip(time[p_peaks], z[p_peaks]))
    n_coord = list(zip(time[n_peaks], z[n_peaks]))

    return p_coord, n_coord


def main():
    time:list = np.arange(-np.pi,np.pi,0.008)   # start,stop,step
    T=np.pi*2
    angular_frequency = (2*np.pi)/T

    excelent_ppg = PPG(1, 0.1999, -1.5161, 0.8186, 0.6303, 1.0225)
    z = calculate_ppg(excelent_ppg, time, angular_frequency)
    p_coord, n_coord = find_peaks(z, time)

    plt.plot(time, z)
    plt.show()


if __name__ == '__main__':
    main()

#ax = plt.axes(projection='3d')
#ax.plot3D(new_x, new_y, new_z, 'gray')
#plt.plot(time[p_peaks], z[p_peaks], 'ro')
#plt.plot(time[n_peaks], z[n_peaks], 'ko')