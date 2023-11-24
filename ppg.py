import matplotlib.pyplot as plt
from dataset import Get_Real_Dataset
from dataclasses import dataclass
import scipy as sp
import numpy as np
import math

time:list = np.arange(0,1,0.008)
T=1
angular_frequency = (2*np.pi)/T

@dataclass
class PPG:
    peak_1: float
    peak_2: float
    theta_1: float
    theta_2: float
    b_1: float
    b_2: float


def build_ppg(ppg:PPG, time:list, angular_frequency:float):
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

    p_peaks, _ = sp.signal.find_peaks(z, prominence=0.05)
    n_peaks, _ = sp.signal.find_peaks(-z)

    p_coord = list(zip(time[p_peaks], z[p_peaks]))
    n_coord = list(zip(time[n_peaks], z[n_peaks]))

    print(p_coord)

    return p_coord, n_coord

def find_last_zero(time: np.array, signal: np.array):
    for i in range(0, len(signal)):
        if signal[i] > 0.1265:
            return time[i]

def get_ppg_constants(z:list):
    p_coord, n_coord = find_peaks(z, time)

    # Theta values calculated using linear conversion from range [0,1] to [-π,π]
    # ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    peak_1  = p_coord[0][1]
    peak_2  = p_coord[1][1]
    theta_1 = ((p_coord[0][0] - 0) / (1 - 0)) * (2*np.pi) - np.pi
    theta_2 = ((p_coord[1][0] - 0) / (1 - 0)) * (2*np.pi) - np.pi
    b_1     = 2 * (p_coord[0][0] - find_last_zero(time, z))
    b_2     = (1 - p_coord[1][0]) * 2
    
    return peak_1, peak_2, theta_1, theta_2, b_1, b_2



def main():
    #synthetic_ppg = PPG(1.014483752077213, 0.20107099910767365, -1.4828317324943823, 0.7791149780902686, 0.528, 0.752)
    
    # get a real PPG signal
    real_ppg, _ = Get_Real_Dataset()

    # get constants for that PPG
    peak_1, peak_2, theta_1, theta_2, b_1, b_2 = get_ppg_constants(real_ppg)
    
    # reconstruct the PPG from the constants we found
    reconstructed_ppg = PPG(peak_1, peak_2, theta_1, theta_2, b_1, b_2)

    # calculate the new z wave
    reconstructed_ppg_wave = build_ppg(reconstructed_ppg, time, angular_frequency)

    plt.plot(time, real_ppg)
    plt.plot(time, reconstructed_ppg_wave)
    plt.show()


if __name__ == '__main__':
    main()

