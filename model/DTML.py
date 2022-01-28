import os
import re
from sqlite3 import DataError
import torch
import configparser
from torch.utils.data import DataLoader 
import torch.nn.functional as F
import torch.optim as optim
import netModel.NetModel as NetModel 

class DTMLNET:

    def __init__(self, configsPath="") -> None:
        self.configs = configparser.ConfigParser()
        self.configsPath = configsPath
        self._loadParams()

        self.dataLoader = None
        self.dataSetInfo  = None
        self.isLoadModel = False
        self.modelFile = None
        self.modelParams = None

        # 初始化配置
        self.batchSize = self._getParam("model","batchSize",0.00002)
        self.epochNum = self._getParam("model", "epoch", 100)
        self.beforeEpoch = -1
        self.epoch = 0
        self.maxEpoch = self.epochNum - 1

        # 读取其他参数
        self.saveStep = self._getParam("env","saveStep",1)
        self.saveNew = self._getParam("env","saveNew",True)

        # 定义设备
        self.device = torch.device(
            f"cuda:{self._getParam('env','GPU','0')}" 
            if torch.cuda.is_available() and self._getParam('env', 'useGPU',False) == 'True' else 'cpu'
        )

        # 初始化网络
        self.mode = "train"
        self.model = None   # 网络模型 
        self.calLoss = None # loss function
        self.optimizer = None

    def buildNet(self, mode="train"):
        self.mode = mode

        if self.dataSetInfo is None:
            raise DataError("Need define data before build a net")

        # 使用的模型
        self.model = NetModel(self.dataSetInfo["shape"][1],self.dataSetInfo["shape"][-1])
        self.model.to(self.device)


        if self.mode == "train":
            # 定义损失函数
            self.lossFuction = self._getParam("model","lossFuction",'MSE')
            self.regularization = self._getParam("model","regularization","L2")        # 正则化的方式
            self.regLambda = self._getParam("model","regLambda",0.001) if self.regularization == "L1" else 0
            if "MSE" == self.lossFuction:
                self.calLoss = torch.nn.MSELoss()
            elif "CE" == self.lossFuction:
                self.calLoss = torch.nn.CrossEntropyLoss()
            elif "NLL" == self.lossFuction:
                self.calLoss = torch.nn.NLLLoss()
            
            # 定义优化器
            self.learningRate = self._getParam("model","learningRate",0.00002)
            self.momentum = self._getParam("model","momentum",0.0)
            self.decay = self._getParam("model","decay",0.9)
            self.optimizerInSet = self._getParam("model","optimizer","RMSprop")
            if "RMSprop" == self.optimizerInSet:
                self.optimizer = optim.RMSprop(self.model.parameters(),lr=self.learningRate, momentum=self.momentum,weight_decay=self.decay)
            elif "SGD" == self.optimizerInSet:
                self.optimizer = optim.SGD(self.model.parameterss(),lr=self.learningRate, momentum=self.momentum,weight_decay=self.decay)
            
    
    def loadData(self, dataSet, shuffle=True):
        self.dataSetInfo ={
            "size":  len(dataSet),
            "shape": dataSet[0].shape
        }
        if self.batchSize is None or self.batchSize == -1:
            self.batchSize = self.dataSetInfo["size"]
        self.dataLoader = DataLoader(dataset=dataSet,batch_size=self.batchSize,shuffle=shuffle)

    def loadModel(self, modelFile):
        self.modelFile = modelFile
        # 是否加载模型
        self.isLoadModel = (self._getParam("env", "continue", False) or self.mode == "test") and os.path.isfile(self.modelFile) 
        if self.isLoadModel:
            self.modelParams = torch.load(self.modelFile)
            self.beforeEpoch = self.modelParams['epoch']
            self.model.load_state_dict(self.modelParams['model_state'])
            self.model.to(self.device)
        else:
            if self.mode == "test":
                raise AttributeError("Model file doesn't exist in test mode.")

        if self.mode == "train":
            if self._getParam("env", "epochContinue", False):
                self.epoch = self.beforeEpoch + 1
                self.maxEpoch = self.beforeEpoch + self.epochNum

            if self.isLoadModel:
                self.optimizer.load_state_dict(self.modelParams["optimizer_state"])

    def _loadParams(self):
        if self.configsPath != "":
            self.configs.read(self.configsPath, encoding="utf-8")
        else:
            self.configs.set()

    def _getParam(self, section, options, default):
        if self.configs.has_option(section, options):
            return self.configs.get(section, options)
        else:
            self.configs.set(section, options, str(default))
            return default
    

    def oneEpochTrain(self, dataloader=None, epochInc=True):
        self.model.train()

        if dataloader is None:
            dataLoader = self.dataLoader

        accumuLoss = 0.0

        for batchI,(inputs, targets) in enumerate(dataLoader):
            # 数据迁移
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # 优化器清零
            self.optimizer.zero_grad()
            # 训练
            outputs = self.model(inputs)

            # 针对L1正则化的罚项计算.
            penalty = 0
            if self.regularization == "L1":
                for p in self.model.parameters():
                    penalty += torch.sum(torch.abs(p))

            # 计算loss
            loss = self.calLoss(outputs, targets) + self.regLambda*penalty
            accumuLoss += loss
            # 反向传播
            loss.backward()
            # 超参数更新
            self.optimizer.step()

        if epochInc:
            self.epoch += 1
        
        return accumuLoss

    def test(self):
        self.model.eval()

        accumuLoss = 0.0
        outputs = None
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.dataLoader):
                # 数据迁移
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # 测试
                outputs = self.model(inputs)
                targets = targets.reshape(targets.size(0), 1, 1, -1).squeeze()
                # 张量相减
                loss = torch.sum(torch.abs(outputs - targets))
                accumuLoss += loss

                print("{i} Loss:", loss)
                print("{i} Accumultate Loss:",accumuLoss)
        
        return accumuLoss, outputs

    def save(self, path="", file_fmt ="checkpoint_%s.ckpt",fmt_params=None):
        if self.saveStep != 0:
            return
        elif self.epoch % (self.saveStep + 1) ==1 or self.epoch==self.maxEpoch:
            if path == "":
                path = "./"
            os.makedirs(path,exist_ok=True)
            
            if not self.saveNew:
                filelist=os.listdir(path)
                for f in filelist:
                    if re.fullmatch(file_fmt.replace('.',r'\.').replace('%s','(.+?)'),f) :
                        os.remove(os.path.join(path, f))
            
            if path[-1] != '/':
                path += '/'
            if fmt_params is None:
                fmt_params = (self.epoch)
            file = path + file_fmt % fmt_params

            torch.save({
                "epoch":self.epoch,
                'model_state':self.model.state_dict(),
                'optimizer_state':self.optimizer.state_dict()
            }, file)
