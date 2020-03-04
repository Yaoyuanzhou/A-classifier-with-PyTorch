import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv
 
'''读取csv文件'''
def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((float(row[2]))) 
        x.append((float(row[1])))
    return x ,y
 
 
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
 
 
plt.figure()
x2,y2=readcsv("D:\code\python_code\PyTorch-YOLOv3-master\otherutils/loss.csv")
plt.plot(x2, y2 ,color='red',linewidth='2', label='Total Loss')

plt.grid(axis='y',c='r',linestyle='--',color='black')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


 
# x,y=readcsv("loss/run_.-tag-alexnet_Validation accuracy.csv")
# plt.plot(x, y, 'blue',linewidth=2,label='AlexNet Accuracy')
# # plt.fill_between(x,y,facecolor='blue',alpha=0.1,label='VGG16 Accuacry')
 
# x1,y1=readcsv("loss/run_.-tag-densenet121_nofreezen_Validation accuracy.csv")
# plt.plot(x1, y1, '.-',color='red',label='DenseNet121 no frozen Accuracy')
 
# x1,y1=readcsv("loss/run_.-tag-alexnet_nofreezen_Validation accuracy.csv")
# plt.plot(x1, y1, '.-.',color='purple',label='AlexNet no frozen Accuracy')
 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
 
plt.ylim(0, 100)
# plt.xlim(0, 120)
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
# plt.title('the trainning ',fontsize=24)
plt.xlabel('Steps',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.legend(fontsize=16)
plt.show()
