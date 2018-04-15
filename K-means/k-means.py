import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd

def KmeansCluster(vectors, noofclusters):
    
    vector_indices = list(range(len(vectors)))
    np.random.shuffle(vector_indices)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        #随机初始化中心点，k个Variable
        centroids = [tf.Variable(vectors[vector_indices[i]])
            for i in range(noofclusters)]
        #重新选择中心点
        centroid_value = tf.placeholder('float64')
        cent_assigns = []
        for c in centroids:
            cent_assigns.append(tf.assign(c,centroid_value))
        
        #每个点的类别 初始值为0
        assignments = [tf.Variable(0) for i in range(len(vectors))]
        #给每个点重新分配类别
        assignment_value = tf.placeholder('int32')
        cluster_assigns = []
        for a in assignments:
            cluster_assigns.append(tf.assign(a,assignment_value))
        
        #根据据中心点的距离重新分配各点
        v1 = tf.placeholder('float',[len(vectors[0])])
        v2 = tf.placeholder('float',[len(vectors[0])])
        distance = tf.sqrt(tf.reduce_sum(tf.pow((v1-v2),2)))
        
        centroid_dist = tf.placeholder('float',[noofclusters])
        cluster_assignment = tf.argmin(centroid_dist,0)
        
        #重新计算中心点
        mean_input = tf.placeholder('float',[None,len(vectors[0])])
        mean_cluster = tf.reduce_mean(mean_input,0)
        
        init = tf.global_variables_initializer()
        
        sess.run(init)
        iteration = 10
        for i in range(iteration):
            #根据据中心点的距离重新分配各点
            for vector_n in range(len(vectors)):
                d = [sess.run(distance,feed_dict = {
                    v1:vectors[vector_n], v2:sess.run(c)})
                        for c in centroids]
                a = sess.run(cluster_assignment,feed_dict = {
                    centroid_dist : d})
                sess.run(cluster_assigns[vector_n],feed_dict = {
                    assignment_value:a}) 
            #最大化步骤，根据聚类的位置重新寻找中心点
            for c in range(noofclusters):
                print(sess.run(assignments))
                assigned_v = [vectors[i] for i in range(len(vectors))
                    if sess.run(assignments[i]) == c]
                new_location = sess.run(mean_cluster,feed_dict = {
                    mean_input:assigned_v})
                sess.run(cent_assigns[c],feed_dict = {
                    centroid_value:new_location})
        
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments
        
sampleNo = 500;#数据数量
mu =3
# 二维正态分布
mu = np.array([[1, 5]])
Sigma = np.array([[1, 0.5], [1.5, 3]])
R = cholesky(Sigma)
srcdata= np.dot(np.random.randn(sampleNo, 2), R) + mu
plt.plot(srcdata[:,0],srcdata[:,1],'bo')
k=4
center,result=KmeansCluster(srcdata,k)
print(center) 
############利用seaborn画图###############

res={"x":[],"y":[],"kmeans_res":[]}
for i in range(len(result)):
    res["x"].append(srcdata[i][0])
    res["y"].append(srcdata[i][1])
    res["kmeans_res"].append(result[i])
pd_res=pd.DataFrame(res)
sns.lmplot("x","y",data=pd_res,fit_reg=False,size=5,hue="kmeans_res")

plt.show()