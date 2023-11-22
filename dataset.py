import matplotlib.pyplot as plt
from mat73 import loadmat 
import numpy as np

def Build_Dataset(FieldName='Subset'):
        Data=loadmat('VitalDB_AAMI_Test_Subset.mat')
        Signals=Data['Subset']['Signals']
        ecg = Signals[1][0]
        ppg = Signals[2][1]
        abp = Signals[3][2]

        #fig, axs = plt.subplots(3,1,figsize=(6,6))
        #axs[0].plot(ecg)
        #plt.plot(ppg[:125])
        #axs[2].plot(abp)
        #plt.show()

        return ppg[:125]


if __name__ == '__main__':
    Build_Dataset()