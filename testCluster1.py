# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
# 生成示例数据
data = {
    'age': [23, 30, 45, 18, 50, 28, 35, 40, 20, 33],
    'score': [85, 92, 78, 88, 95, 83, 87, 80, 85, 89]
}
 
df = pd.DataFrame(data)
 
# 数据预处理
X = df[['age', 'score']]
 
# 设置聚类数量为2
k = 2
 
# 运行K均值聚类算法
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
 
# 获取聚类结果
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
 
# 可视化聚类结果
fig, ax = plt.subplots()
for i in range(k):
    cluster_points = X[labels == i]
    ax.scatter(cluster_points['age'], cluster_points['score'], label='Cluster {}'.format(i+1))
 
ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label='Centroids')
ax.set_xlabel('Age')
ax.set_ylabel('Score')
ax.legend()
plt.savefig("./saveFigs/testCluster.jpg")
plt.show()

