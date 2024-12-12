
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



if 0:
    dfall.to_csv('.\\stca_dataset_for_test.csv',index=False)

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

if 0:
    frameList = dfall.frame.unique()  # 获得每一条frame
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
        
        if i>4:
            break
        
        
    frames = []
    savePath = ".\\saveFig"
    savePath = "./saveFig"
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



frameList = dfall.frame.unique().sort()  # 获得每一条frame
numFrame = len(frameList)
sampleLen = 2*30 #帧率30,2秒的数据为60帧
samplesAll =  {} #字典
for i, frameid in enumerate(frameList):  # 枚举每一个frame，数据库中frameid从9000开始，也就是从5分钟开始

    if i<1.5*sampleLen+1:
        continue
    df_frame =dfall[dfall["frame"]== frameid]#当前时间
    df_frameLast =dfall[dfall["frame"]== frameid-30]#当前时间的前一秒，30frame等于1秒
    trackList =  df_frame.TrackID.unique()  # 获得每一条track
    numTrack = len(trackList)

    
    print(f'i:{i},frameid:{frameid},numFrame:{numFrame}')
    
    #if frameid<9365:
    #    continue
        
    samples = {}#字典
    for j, trackid in enumerate(trackList): # 枚举每一个track
        samplesTmp1  = []
        for k in range(frameid - sampleLen,frameid):#获得当前时刻前面2秒的数据
            df2 = df_frame[df_frame["TrackID"]== j]
            df_pt = df2[df2["lost"]== 0]
            
            df2 = df_frameLast[df_frameLast["TrackID"]== j]
            df_ptl = df2[df2["lost"]== 0]#是说与前一秒的速度比较
            
            if df_pt.empty == True or df_ptl.empty == True:
                continue
            x = df_pt['xmin'].iloc[0]
            y = df_pt['ymin'].iloc[0]
            xl = df_ptl['xmin'].iloc[0]
            yl = df_ptl['ymin'].iloc[0]
            
            vx = x-xl
            vy = y-yl
           
            v = math.sqrt(vx*vx+vy*vy)
            theta = math.atan2(vy,vx)*180/math.pi
             
            samplesTmp1.extend([x,y,v,theta])
            x = df_pt['xmin'].iloc[0]
            y = df_pt['ymin'].iloc[0]
            xl = df_ptl['xmin'].iloc[0]
            yl = df_ptl['ymin'].iloc[0]
            
            vx = x-xl
            vy = y-yl
           
            v = math.sqrt(vx*vx+vy*vy)
            theta = math.atan2(vy,vx)*180/math.pi
             
            samplesTmp1.extend([x,y,v,theta])
    
        #print(samplesTmp1)
        if not samplesTmp1:
            continue
        samples[trackid] = samplesTmp1 # 存当前时刻和前一时刻（2秒），当前轨迹的据
     
    
    samplesAll[frameid] = samples
    #每个时刻都保持一下，免得出问题
    trainSamples = pd.DataFrame(samplesAll)
    
    filename = "./samples/frame%d_2second_samples_%s.csv" %(frameid,datetime.now())
    trainSamples.to_csv(filename,index=False)

print(trainSamples.head(5))


