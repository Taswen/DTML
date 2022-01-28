import sys
from os.path import abspath, join, dirname

from model.dataModel import CASSI, HSI2Grey, HSISingleData
sys.path.insert(0, join(abspath(dirname(__file__)), ''))

from model.DTML import DTMLNET


m = DTMLNET('./config/test_config.ini')
m.loadData(HSISingleData('./data/', CASSI, HSI2Grey))
# m.loadData(HSISingleData('./data/',CASSI))

print("DataSet Length:",m.dataSetInfo["size"])
print("DataSet Shape:",m.dataSetInfo["shape"])

m.buildNet("test")
m.loadModel('./save/checkpoint/')
#TODO 计时
print("-------------------------\n"+
      "End with loss: ",m.test())