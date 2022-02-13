import sys
from os.path import abspath, join, dirname
import time

from model.dataModel import CASSI, HSISingleData, X0FromLS, normalizeByMax, repeatDim
sys.path.insert(0, join(abspath(dirname(__file__)), ''))

from model.DTML import DTMLNET

DATASETNAME="Harvard"
EPOCH="99"

m = DTMLNET('./config/test_config.ini')
m.loadData(HSISingleData(f'./data/Test/{DATASETNAME}48/44.mat', CASSI))
# m.loadData(HSISingleData(f'./data/Test/{DATASETNAME}48/44.mat', CASSI))
# m.loadData(HSISingleData(f'./data/Test/{DATASETNAME}48/44.mat', lambda x:repeatDim(CASSI(x)[0], 2)))
m.loadData(HSISingleData(f'./data/Test/{DATASETNAME}48/44.mat', lambda x:normalizeByMax(CASSI(x)[0])))
m.loadData(HSISingleData(f'./data/Test/{DATASETNAME}48/44.mat', lambda x:normalizeByMax(X0FromLS(*CASSI(x)))))


print("-------------------------------------")
print("Data Set:", DATASETNAME)
print("DataSet Length:", m.dataSetInfo["size"])
print("DataSet Input Shape:", m.dataSetInfo["XShape"])
print("DataSet Target Shape:", m.dataSetInfo["YShape"])
print("-------------------------------------\n")

m.buildNet("test")
m.loadModel(f'./save/checkpoints/{DATASETNAME}/checkpoint_{EPOCH}.ckpt')

beginTime = time.perf_counter()
print(f"Begin at {time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(beginTime))}")
print("-------------------------------------")
print("End with loss: ",m.test())
print("-------------------------------------")
endTime = time.perf_counter()
print(f"End at {time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(endTime))}")
interval = endTime - beginTime
h = interval/3600
m = (interval - h*3600)/60
s = interval - h*3600 - m*60
print("Total Time:", f"{h}h{m}m{s}s")