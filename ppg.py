import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy as sp
import numpy as np
import math

# time vector from -π to +π
time_pi:list = np.arange(-np.pi,np.pi,0.008)
T_pi=np.pi*2
angular_frequency_pi = (2*np.pi)/T_pi

# time vector from 0 to 1
time_1:list = np.arange(0,1,0.008)
T_1=1
angular_frequency_1 = (2*np.pi)/T_1

@dataclass
class PPG:
    peak_1: float
    peak_2: float
    theta_1: float
    theta_2: float
    b_1: float
    b_2: float


def calculate_ppg(ppg:PPG, time:list, angular_frequency:float):
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

def get_ppg_constants(ppg:PPG, z:list):

    p_coord, n_coord = find_peaks(z, time_pi)
    print(f"Positivas:{p_coord}")

    # re-builds the same signal with time going from 0 to 1
    # this is needed so we can find b
    new_z = calculate_ppg(ppg, time_1, angular_frequency_1)
    new_p_coord, new_n_coord = find_peaks(new_z, time_1)
    print(f"Novas Positivas:{new_p_coord}")

    peak_1 = p_coord[0][1]
    peak_2 = p_coord[1][1]
    theta_1 = p_coord[0][0]
    theta_2 = p_coord[1][0]
    b_1 = new_p_coord[0][0] * 2
    b_2 = (1 - new_p_coord[1][0]) * 2
    
    return peak_1, peak_2, theta_1, theta_2, b_1, b_2



def main():
    # generates a perfect synthetic ppg
    synthetic_ppg = PPG(1, 0.1999, -1.5161, 0.8186, 0.6303, 1.0225)
    
    # calculate the z wave
    synthetic_z = calculate_ppg(synthetic_ppg, time_pi, angular_frequency_pi)

    # get constants
    peak_1, peak_2, theta_1, theta_2, b_1, b_2 = get_ppg_constants(synthetic_ppg, synthetic_z)
    
    # reconstruct the PPG from the constants we found
    reconstructed_ppg = PPG(peak_1, peak_2, theta_1, theta_2, b_1, b_2)

    # calculate the new z wave
    reconstructed_ppg_wave = calculate_ppg(reconstructed_ppg, time_pi, angular_frequency_pi)

    plt.plot(time_pi, synthetic_z)
    plt.plot(time_pi, reconstructed_ppg_wave)
    plt.show()


if __name__ == '__main__':
    main()

