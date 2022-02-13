from torch.utils.data import Dataset
import h5py
import numpy as np


def HSI2RGB(input):
    return repeatDim(input, 3)


def HSI2Grey(input):
    """HSI图像加和到一维
    """
    return np.sum(input, axis=2)                        # 将 channel 维度加和


def repeatDim(input, dimNum):
    """一维图像在通道维复制多次
    """
    return np.tile(input,(1,1,dimNum))                  # 扩展通道数量


def normalizeByMax(input):
    """归一化
    """
    avgV = input.mean()
    maxV = input.max() 
    return (input - avgV) / maxV  


def get_rand_mask(h,w,c):
    """生成随机的掩膜 
    """
    mask = np.zeros([h, w, c])                          # 初始化编码孔径
    T = np.round(np.random.rand(int(h/2), int(w/2)))    # 随机分布
    T = np.concatenate([T,T],axis=0)                                    
    T = np.concatenate([T,T],axis=1)                    # 上两步横向和纵向都两倍扩展
    for ch in range(c):
        mask[:,:,ch] = np.roll(T, shift=-ch, axis=0)    # 在height维度进行移位操作
    return mask


def CASSI(input):
    """模拟压缩感知的过程，从真值到压缩过的图像
    """
    # 随机生成掩膜 
    mask = get_rand_mask(*input.shape)
    # 掩膜
    y = np.multiply(input, mask)                        # 使用掩膜进行编码（把 x 和掩膜对应位置相乘）
    y = np.sum(y, axis=2,keepdims=True)                 # 将 channel 维度加和, 但不降维
    return y, mask

def X0FromLS(input, mask):
    c = mask.shape[2]
    mutil_y = np.tile(input,(1, 1, c))                  # 升维到通道数个维度
    return np.multiply(mutil_y, mask)                   # 再次使用掩膜进行编码（把 y2 和掩膜对应位置相乘）

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
