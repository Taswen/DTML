import sys
from os.path import abspath, join, dirname
import time

from model.dataModel import CASSI, HSISingleData, X0FromLS, normalizeByMax, repeatDim
sys.path.insert(0, join(abspath(dirname(__file__)), ''))

from model.DTML import DTMLNET

DATASETNAME="Harvard"

m = DTMLNET('./config/train_config.ini')
# m.loadData(HSISingleData(f'./data/Train/Training_Data_{DATASETNAME}_48.mat', CASSI[0]))
# m.loadData(HSISingleData(f'./data/Train/Training_Data_{DATASETNAME}_48.mat', lambda x:repeatDim(CASSI(x)[0], 2)))
# m.loadData(HSISingleData(f'./data/Train/Training_Data_{DATASETNAME}_48.mat', lambda x:normalizeByMax(CASSI(x)[0])))
m.loadData(HSISingleData(f'./data/Train/Training_Data_{DATASETNAME}_48.mat', lambda x:normalizeByMax(X0FromLS(*CASSI(x)))))

print("-------------------------------------")
print("Data Set:", DATASETNAME)
print("DataSet Length:", m.dataSetInfo["size"])
print("DataSet Input Shape:", m.dataSetInfo["XShape"])
print("DataSet Target Shape:", m.dataSetInfo["YShape"])

m.buildNet()
# m.loadModel(f'./save/checkpoints/{DATASETNAME}/checkpoint_99.ckpt')

print("-------------------------------------")
print("Continue:", m.isLoadModel)
print("From epoch:", m.epoch)
print("To epoch:", m.maxEpoch)
print("Totol epoch:", m.epochNum)
print("Learning Rate:", m.learningRate)
print("-------------------------------------\n")
beginTime = time.time()
print(f"Begin at {time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(beginTime))}")
print("-------------------------------------")

for epoch in range(m.epoch, m.maxEpoch+1):
    loss = m.oneEpochTrain(epoch)
    print(f"Epoch {epoch}: loss:{loss} lr:{m.optimizer.state_dict()['param_groups'][0]['lr']}")
    m.save(f"./save/checkpoints/{DATASETNAME}/")
    m.epoch += 1
print("-------------------------------------")
endTime = time.time()
print(f"End at {time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(endTime))}")
interval = endTime - beginTime
h = interval/3600
m = (interval - h*3600)/60
s = interval - h*3600 - m*60
print("Total Time:", f"{h}h{m}m{s}s")