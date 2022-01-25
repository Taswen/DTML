import sys
from os.path import abspath, join, dirname

from model.dataModel import CASSI, HSI2Grey, HSISingleData
sys.path.insert(0, join(abspath(dirname(__file__)), ''))

from model.DTML import DTMLNET


m = DTMLNET('./config/train_config.ini')
m.loadData(HSISingleData('./data/', CASSI, HSI2Grey)) # 拍平了
# m.loadData(HSISingleData('./data/',CASSI))
m.buildNet()
m.loadModel('./save/checkpoint/')
for epoch in range(m.epoch, m.maxEpoch):
    m.oneEpochTrain()
    m.save()


