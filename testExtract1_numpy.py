
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import math
import time
from datetime import datetime
'''
简单测试读写
'''
file1  = ".\\stanford_campus_dataset\\annotations\\deathCircle\\video0\\annotations.txt"
file1  = './stanford_campus_dataset/annotations/deathCircle/video0/annotations.txt'
f1 = open(file1)
print("Name of the file:", f1.name)
line = f1.readline()
print("Read Line: %s" % (line))



'''
#########################################################
以pandas为工具进行读写
'''

headers = ["TrackID", "xmin", "ymin", "xmax", "ymax","frame", "lost", "occluded", "generated","label"]
dfall = pd.read_csv(file1, sep=' ',names=headers)
print(dfall.head(5))





'''
#########################################################
提取特征，为2秒钟的数据
特征包括:1 位置 2 速度 3 速度方向 4 路径编号
聚类输出 ：属于各个团
分类输出 ：二分法，是否输出处于车队中

'''

trackidcol = 0
xmincol =1
ymincol =2  
xmaxcol = 3
ymaxcol  = 4 
framecol =5
lostcol = 6
occludedcol = 7  
generatedcol =  8  
labelcol = 9

frameList = dfall.frame.unique()  # 获得每一条frame
frameList.sort()
print('framelist:',frameList)
numFrame = frameList.size
sampleLen = 2*30 #帧率30,2秒的数据为60帧
#samplesAll =  {} #字典
datall= dfall.to_numpy()
samplesList =  [] #字典
for i, frameid in enumerate(frameList):  # 枚举每一个frame，数据库中frameid从9000开始，也就是从5分钟开始

    if i<1.5*sampleLen+1:
        continue
    indexT = datall[:,framecol ] == frameid
    dataframe =datall[indexT,:] #当前时间的前一秒，30frame等于1秒
    dataframelast = datall[datall[:,framecol ] == frameid-30,:] 
    trackList = dataframe[:,trackidcol]
    numTrack = len(trackList)
    print(f'i:{i},frameid:{frameid},numFrame:{numFrame},numTrack:{numTrack}')
    #print(dataframe[:,1:5])
    #print(dataframelast[:,1:5])
    
    
    if i>200000:
        break
        
    #samples = {}#字典
    for j, trackid in enumerate(trackList): # 枚举每一个track
        
        samplesTmp1  = []
        for k in range(frameid - sampleLen,frameid):#获得当前时刻前面2秒的数据
            indexT = dataframe[:,trackidcol]== trackid
            df2 = dataframe[indexT,:]
            indexT = df2[:,lostcol]== 0
            df_pt = df2[indexT,:]
            
            indexT = dataframelast[:,trackidcol]== trackid
            df2 = dataframelast[indexT,:]
            indexT = df2[:,lostcol]== 0
            df_ptl = df2[indexT,:]

            if df_ptl.size == 0 or df_pt.size == 0:
                continue
            df_pt = df_pt[0]
            df_ptl = df_ptl[0]
            x = df_pt[xmincol]
            y = df_pt[ymincol]
            xl = df_ptl[xmincol]
            yl = df_ptl[ymincol]
            
            vx = x-xl
            vy = y-yl
           
            v = math.sqrt(vx*vx+vy*vy)
            theta = math.atan2(vy,vx)*180/math.pi
            samplesTmp1.extend([x,y,v,theta])
            
        if not samplesTmp1:
            continue
        #samples[trackid] = samplesTmp1 # 存当前时刻和前一时刻（2秒），当前轨迹的据
        samplesTmp2  = [frameid,trackid]
        samplesTmp2.extend(samplesTmp1) 
        samplesList.append(samplesTmp2)
        
        
    #samplesAll[frameid] = samples
  
    #每个时刻都保存一下，免得出问题
    #if frameid%1000 == 1:
    #    trainSamples = pd.DataFrame(samplesAll)
    #    filename = "./samples/frame:%6d_time:%s.csv" %(frameid,datetime.now())
    #    trainSamples.to_csv(filename)
        
    if frameid%1000 == 1 or i == numFrame-1:
        trainSamples2 = pd.DataFrame(samplesList)
        filename = "./samples/frame:%06d_samples%010d_time:%s.csv" %(frameid,len(samplesList),datetime.now())
        
        head1 = ['frameid','trackid']
        head1.extend(list(range(len(samplesList[0])-2)))
        trainSamples2.to_csv(filename,index=False,header = head1)

print(trainSamples2.head(5))
