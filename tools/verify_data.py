import sys
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath( dirname(__file__)), '../'))
from model.dataModel import CASSI, HSISingleData, X0FromLS, get_rand_mask, normalizeByMax, repeatDim

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

DATASETNAME="Harvard"
# DATASETNAME="ICVL"


import h5py

class showHSI:
    def __init__(self, Xdata, Ydata=None):
        self.Xdata = Xdata
        self.Ydata = Ydata
        if self.Ydata is not None:
            assert Xdata.shape == Ydata.shape
            self.vmin = min(np.min(self.Xdata), np.min(self.Ydata))
            self.vmax = min(np.max(self.Xdata), np.max(self.Ydata))
            self.fig, self.ax = plt.subplots(1, 2)
            self.axX = self.ax[0]
            self.axY = self.ax[1]
        else:
            self.vmin = np.min(self.Xdata)
            self.vmax = np.max(self.Xdata)
            self.fig, self.ax = plt.subplots(1, 1)
            self.axX = self.ax
        
        print(self.vmin,self.vmax)
        
        self.currentDataIndex = 0
        if len(Xdata.shape) == 2:
            self.dataLen = 1
            self.currentXData = Xdata            
        elif len(Xdata.shape) == 3:
            self.dataLen = Xdata.shape[2]
            self.currentXData = Xdata[:,:,self.currentDataIndex]
        else:
            print(f"UnExpect Shape: {Xdata.shape}")
        
        self.Xplt = self.axX.imshow(self.currentXData, cmap="gray", vmin=self.vmin, vmax=self.vmax)
        self.axX.set_title(f"X: {self.currentDataIndex}/{self.dataLen-1}")
        self.fig.colorbar(self.Xplt, ax=self.axX)

        if self.Ydata is not None:
            self.currentYData = Ydata[:,:,self.currentDataIndex]
            self.Yplt = self.axY.imshow(self.currentYData, cmap="gray", vmin=self.vmin, vmax=self.vmax)
            self.axY.set_title(f"Y: {self.currentDataIndex}/{self.dataLen-1}")
            self.fig.colorbar(self.Yplt, ax=self.axY)
        if len(Xdata.shape) == 3:
            plt.subplots_adjust(bottom=0.2)
            self.setButtons()
    
    def show(self):
        plt.show()

    def _next(self, event):
        if self.currentDataIndex < self.dataLen-1:
            self.currentDataIndex += 1
            self.setData()

    def _prev(self, event):
        if self.currentDataIndex > 0:
            self.currentDataIndex -= 1
            self.setData()

    def setData(self):
        self.currentXData = self.Xdata[:,:,self.currentDataIndex]
        self.Xplt.set_data(self.currentXData)
        self.axX.set_title(f"X: {self.currentDataIndex}/{self.dataLen-1}")
        if self.Ydata is not None:
            self.currentYData = self.Ydata[:,:,self.currentDataIndex]
            self.Yplt.set_data(self.currentYData)
            self.axY.set_title(f"Y: {self.currentDataIndex}/{self.dataLen-1}")
        plt.draw()  

    def setButtons(self):
        ax_prev = plt.axes([0.41, 0.05, 0.07, 0.075])
        ax_next = plt.axes([0.51, 0.05, 0.07, 0.075])
        # btn_prev = Button(ax_prev,"<")
        self.prev_button = Button(ax_prev,"<")
        self.prev_button.on_clicked(self._prev)
        self.next_button = Button(ax_next,">")
        self.next_button.on_clicked(self._next)
                    

def cutAndSave(file, num = 4):
    """ 从大的训练数据上切下来一部分以进行观察
    """
    dataset = HSISingleData(file, (3, 0, 1, 2), lambda x:X0FromLS(*CASSI(x), labelName="label"))
    cal_x, orl_y = np.zeros((num, 48, 48, 31)), np.zeros((num, 48, 48, 31))
    for i in range(4):
        cal_x[i], orl_y[i] = dataset[i]
        # print("X:\n", cal_x[i])
        # print("Y:\n", orl_y[i])
        # print("\n")
    np.save(f"save/cutData/cal_x_{num}", cal_x)
    np.save(f"save/cutData/orl_y_{num}", orl_y)

def getCut(num = 4):
    """ 读取切下来的数据
    """
    return np.load(f"./save/cutData/cal_x_{num}.npy"), np.load(f"save/cutData/orl_y_{num}.npy")


if __name__ == "__main__":
    # 测试showHSI的可用性
    # arr3 = np.ones((10,10,3))
    # arr = np.arange(300.)
    # arr3 = arr.reshape((10,10,3),order='F')/450.0
    # arr3y = np.ones((10,10,3))
    # arr = np.arange(150,450.)
    # arr3y = arr.reshape((10,10,3),order='F')/450/0
    # showHSI(arr3,arr3y).show()

    # 测试直接读取原始48*48*31的数据
    # cutAndSave(f'./data/Train/Training_Data_{DATASETNAME}_48.mat')
    # data   = h5py.File(file, 'r') 
    # labels = data['label']                        # 获取其中数据标签
    # del data
    # new_labels = np.transpose(labels, (0, 3, 2, 1))   # 调整维度的顺序（batch_size，h,w,c）
    # one_label = new_labels[0]
    # showHSI(one_label).show()

    # 测试生成的mask
    # mask = get_rand_mask(48, 48, 31)
    # showHSI(mask).show()

    # 测试使用最小二乘后的 X0
    # file = f'./data/Train/Training_Data_{DATASETNAME}_48.mat'
    # x,y = getCut()
    # showHSI(x[0], y[0]).show()

    # file = f'./data/Train/Training_Data_{DATASETNAME}_48.mat'
    # data = HSISingleData(file, (3, 0, 1, 2), labelName="label")
    # 测试集中512*512的原始数据
    file = f'./data/Test/{DATASETNAME}48/44.mat'
    data = HSISingleData(file, None, lambda x:X0FromLS(*CASSI(x)), labelName="hyper_image")
    showHSI(data[0][0], data[0][1]).show()
    
