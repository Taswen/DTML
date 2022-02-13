import torch
import torch.nn.functional as F


class NetModel(torch.nn.Module):
    def __init__(self, n, c) -> None:
        torch.nn.Module.__init__(self)
        # FC1(输入层到隐含层)
        self.fc1 = torch.nn.Linear(c*(n**2), c*(n**2))
        # FC2
        self.fc2 = torch.nn.Linear(c*(n**2), n*n)
        # FC3
        self.fc3 = torch.nn.Linear(n*n, n*n)

        # Convolution layers
        self.C1 = torch.nn.Conv2d(1,64, kernel_size=5, padding=2)
        self.C2 = torch.nn.Conv2d(64,64, kernel_size=5, padding=2)
        self.DeC = torch.nn.Conv2d(64,31, kernel_size=7, padding=3)

    def forward(self, x):

        batchSize,h,w = x.size(0), x.size(1), x.size(2)
        # 除了batch维,其他连成一维向量，
        x = x.reshape(batchSize, 1, 1, -1).squeeze()

        # 全连接
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        # 重新整型为二维的
        x = x.reshape(batchSize,-1,h,w)

        # 卷积操作
        x = F.relu(self.C1(x))
        x = F.relu(self.C2(x))
        x = F.relu(self.DeC(x))

        # 以便计算loss
        return x.reshape(batchSize, -1)

