import sys
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath( dirname(__file__)), '../'))
from model.dataModel import HSISingleData

import h5py
import numpy as np
DATASETNAME="Harvard"


def cuthalf():
    rfile = f'./data/Train/Training_Data_{DATASETNAME}_48.mat'
    wfile = f'./data/Train/Training_Data_{DATASETNAME}_24.mat'
    data = h5py.File(rfile, 'r')
    label = data['label']
    n, c, w, h = label.shape
    halfW = w//2
    halfH = h//2
    lu = label[:, :, 0:halfW, 0:halfH]
    ru = label[:, :, 0:halfW, halfH:h]
    ld = label[:, :, halfW:w, 0:halfH]
    rd = label[:, :, halfW:w, halfH:h]
    # Version 2
    nn = np.zeros((len(lu) * 4, c , halfW, halfH))
    for i in range(0,len(lu)):
        nn[4 * i] = lu[i]
        nn[4 * i + 1] = ru[i]
        nn[4 * i + 2] = ld[i]
        nn[4 * i + 3] = rd[i]
        if i % 400 == 0:
            print(i)
    # Version 1
    # nn = np.stack([lu[0],ru[0],ld[0],rd[0]])
    # for i in range(1,len(lu)):
    #     nn = np.concatenate([nn, np.stack([lu[i], ru[i], ld[i], rd[i]])], axis=0)
    with h5py.File(wfile, 'w') as new_data:
        new_data.create_dataset('label', data=nn, compression='gzip')
        new_data.create_dataset('set', data=np.expand_dims(np.repeat(data['set'], 4),axis=1) , compression='gzip')
        # new_data["label"] = nn
        # new_data["set"] = np.expand_dims(np.repeat(data['set'], 4),axis=1) 
        new_data.flush()

def cut(num = 4):
    rfile = f'./data/Train/Training_Data_{DATASETNAME}_48.mat'
    wfile = f'./data/Train/Training_Data_{DATASETNAME}_test.mat'
    data = h5py.File(rfile, 'r')
    label = data['label']
    new_data = h5py.File(wfile, 'w')
    # new_data.create_dataset("label",dtype=label.dtype)
    # new_data.create_dataset("set",dtype=data['set'].dtype)

    # new_data.create_dataset("PHITPHI",dtype=data['PHITPHI'].dtype)
    # new_data.create_dataset("SZ",dtype=data['SZ'].dtype)
    new_data.create_dataset('label', data=data['label'][0:num], compression='gzip')
    new_data.create_dataset('set', data=data['set'][0:num], compression='gzip')
    # new_data["label"] = data['label'][0:num]
    # new_data["set"] = data['set'][0:num]
    new_data.flush()

if __name__ == "__main__":
    # cuthalf()
    # cut()
    # rfile = f'./data/Train/Training_Data_{DATASETNAME}_48.mat'
    # data = h5py.File(rfile, 'r')
    # label = data['set']
    # print(np.array(label))
    dataset = HSISingleData(f'./data/Train/Training_Data_{DATASETNAME}_24.mat', (0, 3, 2, 1))
    print(len(dataset))
    print(dataset[0][0].shape)