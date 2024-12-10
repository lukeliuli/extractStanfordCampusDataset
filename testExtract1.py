
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import math
'''
简单测试读写
'''
file1  = '.\\stanford_campus_dataset\\annotations\\deathCircle\\video0\\annotations.txt'
f1 = open(file1)
print("Name of the file:", f1.name)
line = f1.readline()
print("Read Line: %s" % (line))

'''
#########################################################
以pandas为工具进行读写
'''

headers = ["TrackID", "xmin", "ymin", "xmax", "ymax","frame", "lost", "occluded", "generated","label"]
df = pd.read_csv(file1, sep=' ',names=headers)
print(df.head(5))
df.to_csv('.\\stca_dataset_for_test.csv',index=False)

'''
#########################################################
以track为关键点进行查看和统计
'''
trackList = df.TrackID.unique()  # 获得每一条track
numTrack = len(trackList)

trackList =[]
for i, id in enumerate(trackList):  # 枚举每一个track
    #print("\ntrackIndex is %d,nameID is %s" % (i, id))
    dfTR =df[df["TrackID"]== id]
    xcenter  = (dfTR['xmin']+dfTR["xmax"])/2
    ycenter  = (dfTR['ymin']+dfTR["ymax"])/2
    xcenter = xcenter.to_numpy()
    ycenter = ycenter.to_numpy()
       
    title = "index:%d,trackid:%d" %(i,id)

   
    plt.plot(xcenter,ycenter)
    plt.title(title)
    plt.show()
    break




'''
#########################################################
以frame(时间)为关键点进行查看和统计

'''


frameList = df.frame.unique()  # 获得每一条frame
numFrame = len(frameList)

for i, frameid in enumerate(frameList):  # 枚举每一个frame
    df_frame =df[df["frame"]== frameid]
    trackList =  df_frame.TrackID.unique()  # 获得每一条track
    numTrack = len(trackList)

    for j, trackid in enumerate(trackList):  # 枚举每一个track
        df_track = df_frame[(df_frame["TrackID"]== trackid) & (df_frame["lost"]== 0) & (df_frame["generated"]== 0 )]
        df_track = df_frame[(df_frame["TrackID"]== trackid) & (df_frame["lost"]== 0)& (df_frame["label"]== "Biker")]
        df_track = df_frame[(df_frame["TrackID"]== trackid) & (df_frame["lost"]== 0)]
        xcenter  = (df_track['xmin']+df_track["xmax"])/2
        ycenter  = (df_track['ymin']+df_track["ymax"])/2
       
        xcenter = xcenter.to_numpy()
        ycenter = ycenter.to_numpy()
        #plt.plot(xcenter,ycenter,'.')

        x = df_track['xmin'].to_numpy()
        y = df_track['ymin'].to_numpy()
        plt.plot(x,y,'.')

    title = "frameindex:%d,frameid:%d,trackNum:%d" %(i,frameid,numTrack)
    plt.plot(xcenter,ycenter,'o')
    plt.title(title)
    plt.savefig(".\\saveFig\\test_frameIndex%4d.jpg" %i)
    #plt.show()
    
    if i>30:
        break
    
    
frames = []

for imagename in os.listdir('.\\saveFig'):
    fname= '.\\saveFig\\'+imagename
    frames.append(imageio.imread(fname))

imageio.mimsave(".\\res.gif", frames, 'GIF', duration=2) 

'''
#########################################################
查看图像

'''

fname = '.\\test1\\test1_ref.jpg'
ref = imageio.imread(fname)
fname = '.\\test1\\test1_mask.png'
mask = imageio.imread(fname)

plt.imshow(ref)
#plt.gca().invert_yaxis()
plt.show()



'''
#########################################################
提取特征，为2秒钟的数据
特征包括:1 位置 2 速度 3 速度方向 4 路径编号
聚类输出 ：属于各个团
分类输出 ：二分法，是否输出处于车队中

'''



frameList = df.frame.unique()  # 获得每一条frame
numFrame = len(frameList)
sampleLen = 2*30 #帧率30,2秒的数据为60帧
sampleAll = []
for i, frameid in enumerate(frameList):  # 枚举每一个frame，数据库中frameid从9000开始，也就是从5分钟开始

    if i<sampleLen:
        continue
    df_frame =df[df["frame"]== frameid]
    trackList =  df_frame.TrackID.unique()  # 获得每一条track
    numTrack = len(trackList)

    samples = []
    for j, trackid in enumerate(trackList):  # 枚举每一个track
        samplesTmp1  = []
        for k in range(frameid - samplelen,frameid) #获得当前时刻前面2秒的数据
            df_pt =df[(df["frame"]== k) & (df["TrackID"]== j) ]
            df_ptl =df[(df["frame"]== k-1) & (df["TrackID"]== j) ]
            
            
            x = df_pt['xmin'].to_numpy()
            y = df_pt['ymin'].to_numpy()
            xl = df_ptl['xmin'].to_numpy()
            yl = df_ptl['ymin'].to_numpy()
            vx = x-xl
            vy = y-xl
            v = vx*vx+vy*vy
            theta = math.atan(vy,vx)*180/pi

            samplesTmp1.extend([x,y,v,theta])
    
        samples[trackid] = samplesTmp1 # 存当前时刻和前一时刻（2秒），当前轨迹的据
    samplesAll[frameid] = samples[j]




