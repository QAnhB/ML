from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

X = [[i] for i in [1,2,9,12,20]]

Z = linkage(X, 'single')
fig = plt.figure(figsize=(25,10))
dendrogram(Z)
plt.show()

H = linkage(X,'complete')
fig1 = plt.figure(figsize=(25,10))
dendrogram(H)
plt.show()
