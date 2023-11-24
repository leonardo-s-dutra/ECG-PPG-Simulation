from mat73 import loadmat 
import numpy as np

def Get_Real_Dataset(FieldName='Subset'):
        Data=loadmat('VitalDB_AAMI_Test_Subset.mat')
        Signals=Data['Subset']['Signals']
        ecg = Signals[1][0]
        ppg = Signals[2][1]

        return ppg[:125], ecg[70:195]


if __name__ == '__main__':
    Get_Real_Dataset()