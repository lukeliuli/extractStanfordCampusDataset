# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import imageio


fname = './samples/frame.csv'
datapd = pd.read_csv(fname)
print(datapd.iloc[0]) 
#datanp = datapd.to_numpy()
X = datapd[datapd['frameid']==91].to_numpy()
X1 = X[:,2:]
print(X.shape)

my_minmax_scaler = MinMaxScaler(feature_range=(0, 1))
X = my_minmax_scaler.fit_transform(X1)
################################################################
'''
kmeans
'''
if 0:
# 设置聚类数量为2
    k = 2
    # 运行K均值聚类算法
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    # 获取聚类结果
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    X = X[:,[0,1]]

    for i in range(k):
        cluster_points = X[labels == i,:]
        plt.scatter(cluster_points[:,0], cluster_points[:,1], label='Cluster {}'.format(i))


    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label='Centroids')
    plt.savefig("./saveFigs/testCluster.jpg")

################################################################
'''
DBSCAN
'''
from sklearn.cluster import DBSCAN
if 0:
# 设置聚类数量为2
   
   
    dbscan = DBSCAN(eps=0.5, min_samples=1)
    labels = dbscan.fit_predict(X)
    print(labels)
    # 可视化结果
    X = X[:,[0,1]]
    plt.scatter(X[:, 0], X[:, 1], c=labels)


    plt.savefig("./saveFigs/testCluster.jpg")
    
    
################################################################
'''
AffinityPropagation
'''

from sklearn.cluster import AffinityPropagation
if 1:
# 设置聚类数量为2
   
   
    model = AffinityPropagation(damping=0.9)
    labels = model.fit_predict(X)
    print(labels)
    # 可视化结果
    X = X1[:,[0,1]]
    clusters = np.unique(labels)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
    # 获取此群集的示例的行索引
        row_ix = np.where(labels == cluster)
       # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1])

    fname = './test1/test1_ref.jpg'
    ref = imageio.imread(fname)
    plt.imshow(ref)

    plt.savefig("./saveFigs/testClusterAffinityPropagation.jpg")

    
################################################################
'''
MeanShift
'''
import imageio
from sklearn.cluster import MeanShift
if 0:
# 设置聚类数量为2
   
   
    model = MeanShift()
    labels = model.fit_predict(X)
    print(labels)
    # 可视化结果
    X = X[:,[0,1]]
    clusters = np.unique(labels)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
    # 获取此群集的示例的行索引
        row_ix = np.where(labels == cluster)
       # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
        
    fname = './test1/test1_ref.jpg'
    ref = imageio.imread(fname)
    fname = './test1/test1_mask.png'
    mask = imageio.imread(fname)

    plt.imshow(ref)



    plt.savefig("./saveFigs/testClusterMeanShift.jpg")
    
    
################################################################
'''
scipy.cluster.hierarchy as sch
'''
import imageio
import scipy.cluster.hierarchy as sch
if 1:
# 设置聚类数量为2
   
    Z=sch.linkage(X,  # 数据
              method="single",
              metric="euclidean")
    
        
        
    T=sch.fcluster(Z,criterion="maxclust",t=10)  # 根据给定的类数t，创建Z的聚类
    
    print(T)
    
    labels = T
    X = X[:,[0,1]]
    clusters = np.unique(labels)
    # 为每个群集的样本创建散点图
    for cluster in clusters:
    # 获取此群集的示例的行索引
        row_ix = np.where(labels == cluster)
       # 创建这些样本的散布
        plt.scatter(X[row_ix, 0], X[row_ix, 1])
        
        
    #plt.scatter(circle_x[:,0],circle_x[:,1],c=T)
    #plt.show()
    
    fname = './test1/test1_ref.jpg'
    ref = imageio.imread(fname)
    fname = './test1/test1_mask.png'
    mask = imageio.imread(fname)

    plt.imshow(ref)



    plt.savefig("./saveFigs/testClusterHierarchy.jpg")
    
    plt.close()
    
    sch.dendrogram(Z,labels=range(0,X.shape[0]))
    plt.savefig("./saveFigs/testClusterdendrogram.jpg")