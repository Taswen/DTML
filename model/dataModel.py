from torch.utils.data import Dataset
import h5py
import numpy as np


def HSI2RGB(input):
    #TODO
    pass


def HSI2Grey(input):
    return np.sum(input, axis=2)                        # 将 channel 维度加和


def CASSI(input):
    h,w,c = input.shape
    # 随机生成掩膜 
    mask = np.zeros([h, w, c])                          # 初始化编码孔径
    T = np.round(np.random.rand(int(h/2), int(w/2)))    # 随机分布
    T = np.concatenate([T,T],axis=0)                                    
    T = np.concatenate([T,T],axis=1)                    # 上两步横向和纵向都两倍扩展
    for ch in range(c):
        mask[:,:,ch] = np.roll(T, shift=-ch, axis=0)    # 在height维度进行移位操作
    # 掩膜
    y = np.multiply(input, mask)                        # 使用掩膜进行编码（把 x 和掩膜对应位置相乘）
    y = np.sum(y, axis=2)                               # 将 channel 维度加和
    return y





class HSISingleData(Dataset):
    def __init__(self, file, transformX=lambda x:x, transformY=lambda x:x):
        data   = h5py.File(file,'r') 
        labels = data['label']                        # 获取其中数据标签
        del data
        
        self.labels = np.transpose(labels, (0, 3, 2, 1))   # 调整维度的顺序（batch_size，h,w,c）
        self.transformX = transformX
        self.transformY = transformY
        self.len = labels.shape[0]

    def __getitem__(self ,index):
        # input, target
        return self.transformX(self.labels[index]), self.transformY(self.labels[index])
        
    
    def __len__(self):
        return self.len
