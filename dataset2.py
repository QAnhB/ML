import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt

"""*upload file"""

df = pd.read_csv('D:\Downloads\ML\Raisin_Dataset.csv')

"""*file description"""

print(df)

df.head()

features = df.keys()
print(features)

df.describe()

"""*split dataset into 2 table: one with class, the other with remaining attributes"""

X = df.drop('Class', axis = 1)
Y = df['Class']
print(X)
print(Y)

"""*calculate mean/ average of a given dataset"""

X.mean()

"""*calculate variance:
low variance: data are clustered together, not spread apart widely
high variace: data muc more spread apart from average value
"""

X.var()

"""*calculate covariance:
cov = 0: uncorrelated, no trend (many Y with one X or many X with one Y)
cov > 0: positively correlated, pos trends ( X increases -> Y increases)
cov < 0: negatively correlated, negative trends (X increases -> Y decreases)
computational stepping stone for correlation
"""

X.cov()

"""*Calculate correlation"""

X_corr = X.corr()
print(X_corr)

"""*take upper triangle of correlation matrix and sorted the value in ascending to dertermine the couple that has the highest correlation (avoid duplicate data)"""

upper_tri = X_corr.where(np.triu(np.ones(X_corr.shape), k = 1).astype(bool))
print(upper_tri)

drop_null = upper_tri.unstack().dropna()
sorted_matrix = drop_null.sort_values()
print(sorted_matrix)

"""*Standardlising the features"""

scaled_data = StandardScaler().fit_transform(X)
scaled_data

"""*pca projection to 2D"""

pca = PCA(n_components = 2)
pca.fit(scaled_data)
pca_data = pca.fit_transform(scaled_data)
pca_data.shape

pca_data

"""*calculation variance ratio"""

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

"""*Scree plot: check if pca is working well on dataset or not"""

per_var = np.round(explained_variance*100, decimals = 1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(labels, height = per_var)
plt.xlabel('Principal components')
plt.ylabel('Percentage of explained variance')
plt.title('Scree Plot')
plt.show()

"""*create a DataFrame that will have the principal component values for all 569 samples"""

pca_Df = pd.DataFrame(data = pca_data, columns = ['PC1', 'PC2'])
pca_Df.head()

"""*combine pca table with class table"""

final_Df = pd.concat([pca_Df, df['Class']], axis = 1)
final_Df

"""*visulize PCA in 2D"""

plt.figure()
plt.figure(figsize = (10,10))
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 14)
plt.xlabel('PC1', fontsize = 20)
plt.ylabel('PC2', fontsize = 20)
plt.title('PCA graph', fontsize=20)
targets = ['Kecimen', 'Besni']
colors = ['r', 'g']
for target, color in zip(targets,colors):
  indicesToKeep = final_Df['Class'] == target
  plt.scatter(final_Df.loc[indicesToKeep, 'PC1'], final_Df.loc[indicesToKeep, 'PC2']
              , c = color, s = 50)
  
plt.legend(targets, prop = {'size':15})
plt.show()

"""*Increase number of PC used"""

pca1 = PCA(n_components = 3)
pca1.fit(scaled_data)
pca_data1 = pca1.transform(scaled_data)
pca_data1.shape


explained_variance1 = pca1.explained_variance_ratio_
print(explained_variance1)

per_var1 = np.round(explained_variance1*100,decimals=1)
#create labels for scree plot
labels1 = ['PC' + str(x) for x in range(1,len(per_var1) + 1)]
#creating a bar plot
plt.bar(labels1, height = per_var1)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal component')
plt.title('Scree Plot')
plt.show()
