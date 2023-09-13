""" HAC: start with one cluster, individual item in its own cluster
and iteratively merge clusters until all the items belong to one cluster
+ bottom up approach is followed to merge the clusters together
+ dendrograms are pictorially used to represent the HAC
Technique: single-nearest distance orr single linkage
-> distance between the closet members of two clusters
Dendrogram: a tree like structure which represents hierarchical technique
+ Leaf-Individual
+ Root - One cluster
A cluster at level 1, is the merge of its child cluster at level i + 1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, complete

x = np.array([1,2,9,12,20])
y = [0,0,0,0,0]
plt.scatter(x,y)
plt.show()
data = list(zip(x,y)) #return a list of iterator
#zip(): create an iterator that produces tuples of the form (x, y)
#take values from x and the take value from y (1,0), (2,1), (9,2),...

#compute single linkage
single_data = linkage(data, method='single', metric='euclidean')
dendrogram(single_data)
plt.show()

#compute complete linkage using euclidean distance, visualize using dendrogram\
complete_data = linkage(data, method='complete', metric='euclidean')
dendrogram(complete_data)
plt.show()