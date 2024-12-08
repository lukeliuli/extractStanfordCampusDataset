# 测试Delaunay，Voronoi,scipy.spatial
from scipy.spatial import Delaunay
import numpy as np
points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
tri = Delaunay(points)

import matplotlib.pyplot as plt
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')

print(tri.simplices)
print(tri.neighbors[0])
plt.show()


from scipy.spatial import ConvexHull
rng = np.random.default_rng()
points = rng.random((10, 2))*2  # 30 random points in 2-D
hull = ConvexHull(points)

plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    print('hull.simplices',simplex)
    plt.plot(points[simplex,0], points[simplex,1], 'k-')



###############################################################
from scipy.spatial import KDTree
tree = KDTree(points)
qpt = [0.1,0.53]
dist,index=tree.query(qpt)
print(index)
plt.title('KDTree')
plt.plot(qpt[0], qpt[1], 'h')
plt.plot(points[index,0], points[index,1], 'x')
plt.title('KDTree')
plt.show()

###############################################################


tri = Delaunay(points)
dindex = tri.find_simplex(qpt)
if dindex >=0:
    plt.title('Delaunay')
    pindex=tri.simplices[dindex]
    plt.plot(points[pindex,0], points[pindex,1], 'x')

    plt.triplot(points[:,0], points[:,1], tri.simplices)
    plt.plot(qpt[0], qpt[1], 'h')
    plt.show()
else:
    print("qpt is not in the Delaunay simplices")
###############################################################

from scipy.spatial import Voronoi,voronoi_plot_2d
vor = Voronoi(points)
voronoi_plot_2d(vor)
print('vor.point_region',vor.point_region)
plt.show()


plt.plot(points[:,0], points[:,1], 'o')
plt.plot(vor.vertices[:,0], vor.vertices[:,1], 'o')

for simplex in vor.ridge_vertices:
    simplex = np.asarray(simplex)
    print('vor.ridge_vertices',simplex)
    if np.all(simplex >= 0):
        plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')

#plt.title('vor.ridge_vertices')
#plt.show()


#vor.ridge_points表示每条 Voronoi 脊线（ridge line）附近的点（points）的索引，即每条脊线的控制点。
plt.plot(points[:,0], points[:,1], 'o')
plt.plot(vor.vertices[:,0], vor.vertices[:,1], 'o')
for simplex in vor.ridge_points:
    simplex = np.asarray(simplex)
    print('vor.ridge_points',simplex)
    if np.all(simplex >= 0):
        plt.plot(points[simplex, 0], points[simplex, 1], '+',color="blue")
plt.title('vor.ridge_points')
plt.show()

