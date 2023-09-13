from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

X = [[i] for i in [1,2,9,12,20]]

H = linkage(X,method='single',metric='euclidean')
fig1 = plt.figure(figsize=(10,10))
dendrogram(H)
plt.show()

H1 = linkage(X,method='complete',metric='euclidean')
fig2 = plt.figure(figsize=(10,10))
dendrogram(H1)
plt.show()
