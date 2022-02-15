import sys
from os.path import abspath, join, dirname
import time

from model.dataModel import CASSI, HSISingleData, X0FromLS, normalizeByMax, repeatDim
sys.path.insert(0, join(abspath(dirname(__file__)), ''))

from model.DTML import DTMLNET

DATASETNAME="Harvard"
BATCHSIZE=48
EPOCH="99"

datafile = f'./data/Test/{DATASETNAME}{BATCHSIZE}/44.mat'

m = DTMLNET('./config/test_config.ini')
# m.loadData(HSISingleData(datafile, (3, 0, 1, 2), CASSI, labelName='patch_image'))
# m.loadData(HSISingleData(datafile, (3, 0, 1, 2), CASSI))
# m.loadData(HSISingleData(datafile, (3, 0, 1, 2), lambda x:repeatDim(CASSI(x)[0], 2), labelName='patch_image'))
# m.loadData(HSISingleData(datafile, (3, 0, 1, 2), lambda x:normalizeByMax(CASSI(x)[0]), labelName='patch_image'))
m.loadData(HSISingleData(datafile, (3, 0, 1, 2), lambda x:normalizeByMax(X0FromLS(*CASSI(x))), labelName='patch_image'))


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