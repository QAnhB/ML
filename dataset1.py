import pandas as pd

df = pd.read_csv('D:\Downloads\ML\dataR2.csv')

print(df)

df.head()

df.keys()

df.describe()

#drop(): removes the specified row or column
#remove column: specifying the column axis (axis = 'columns')
#remove row: specifying the row axis(axis = 'index')
X = df.drop('Classification', axis = 1)
print(X)

Y = df['Classification']
print(Y)

#used to calculate mean/average of a given dataset
#=sum of data divided number of value
X.mean()

#variance: measure the spread of random data from its mean
#low variance: data are clustered together, not spread apart widely
#high variace: data muc more spread apart from average value
X.var()

#covariance indicates the level to which two variables vary together, classsify
#three types of relationships
#cov = 0: uncorrelated, no trend (many Y with one X or many X with one Y)
#cov > 0: positively correlated, pos trends ( X increases -> Y increases)
#cov < 0: negatively correlated, negative trends (X increases -> Y decreases)
#computational stepping stone for correlation
X.cov()

#corr_max = 1: when a straight line with a pos slope can go through the center of every data point
#corr = -1: neg slope, strong relationship
#smaller p-value, the more confidence we can have in the guesses we make
#-> correlation quantifies the strength of relationships
#weak -> small corr, moderate -> moderate corr, strong -> large corr
# far slope -> near slope -> nearer slope
X_corr = X.corr()
print(X_corr)

import numpy as np

upper_tri = X_corr.where(np.triu(np.ones(X_corr.shape),k=1).astype(bool))
print(upper_tri)
#np.triu: upper triangle 
#np.tril: lower triangle
#create a boolean matrix with same size as our corr matrix. 
#the boolean matrix will have True value on upper and False value on lower
#then use the boolean matrix with True on upper to extract upper using where()
#where() return a dataframe of original size but with NaN values on lower

#unstack(): reshapes the given dataframe by converting the row index to a column label -> return a transposed DataFrame
#dropna(): remove rows that contain null value
drop_null = upper_tri.unstack().dropna()
#sorted_value(): sorts data in ascending or descending order 
sorted_matrix = drop_null.sort_values()
print(sorted_matrix)

#split the dataset into 2 parts: where you use part of the sample for training and remaing for testing
#reason: want to use the samples that the model has not seen before for testing the model
#-> better accuracy of the model
# 80%: train, 20%:test
# random=2: reproduce a particular code
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

#PCA performs best with a normalized feature set.
#peform standard scalar normalization to normalize our feature set.
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

scaled_data = preprocessing.scale(X)
scaled_data

pca = PCA(n_components = 2)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
pca_data.shape

print(pca)

import matplotlib.pyplot as plt

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

#scree plot: check if pca is working well on dataset or not 
#need to generate an array of the names. (ex: PC1,PC2)
#calculate the percentage of variation that each principal component accounts for
#np.round(): round an array to the given number of decimals
per_var = np.round(explained_variance*100,decimals=1)
#create labels for scree plot
labels = ['PC' + str(x) for x in range(1,len(per_var) + 1)]
#creating a bar plot
plt.bar(labels, height = per_var)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal component')
plt.title('Scree Plot')
plt.show()

print(per_var)

pca_Df = pd.DataFrame(data = pca_data,columns = ['PC1', 'PC2'])
pca_Df.head()

#pd.DataFrame(data, index,columns)

final_Df = pd.concat([pca_Df, df[['Classification']]], axis = 1)
final_Df

final_Df['Classification'].replace(1, 'Healthy controls', inplace = True)
final_Df['Classification'].replace(2,'Patient', inplace = True)
final_Df.head()

final_Df.tail()

plt.figure()

plt.figure()
plt.figure(figsize = (10,10))
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 14)
plt.xlabel('PC1', fontsize = 20)
plt.ylabel('PC2', fontsize = 20)
plt.title('PCA graph', fontsize=20)
targets = ['Healthy controls', 'Patient']
colors = ['r', 'g']
for target, color in zip(targets,colors):
  indicesToKeep = final_Df['Classification'] == target
  plt.scatter(final_Df.loc[indicesToKeep, 'PC1'], final_Df.loc[indicesToKeep, 'PC2']
              , c = color, s = 50)
  
plt.legend(targets, prop = {'size':15})
plt.show()

pca1 = PCA(n_components = 5)
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

pca2 = PCA(n_components = 6)
pca2.fit(scaled_data)
pca_data2 = pca2.transform(scaled_data)
pca_data2.shape

explained_variance2 = pca2.explained_variance_ratio_
print(explained_variance2)

per_var2 = np.round(explained_variance2*100,decimals=1)
#create labels for scree plot
labels2 = ['PC' + str(x) for x in range(1,len(per_var2) + 1)]
#creating a bar plot
plt.bar(labels2, height = per_var2)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal component')
plt.title('Scree Plot')
plt.show()