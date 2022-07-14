#!/usr/bin/env python
# coding: utf-8

# # Clustering using k-means algorithm

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib as mpl

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


# In[6]:


# retrieving dataset using pandas
file_path = "C:\\Users\\izzug\\Desktop\\Clustering\\annotations.csv" #absolute path
df = pd.read_csv(file_path)
df.head() 


# In[ ]:


# organizing data 

df_updated = []
for i in range(len(df)):
    x = df.iloc[i]['x y']
    x = list(map(eval, x.split(' ')))
    df_updated.append(x)
df_new = pd.DataFrame (df_updated, columns = ['x', 'y'])
df_new


# In[ ]:


#testing
train = df_new.sample(frac=0.8,random_state=200)
test = df_new.drop(train.index)
print(len(train),len(test))


# In[ ]:


train.values


# In[ ]:


#clustering
model = KMeans(3) #number of clusters
model.fit(df_new)


# In[ ]:


#clusters (result of clustering)
identified_clusters = model.fit_predict(df_new)
identified_clusters


# In[ ]:


#scatterplot of the result
data_with_clusters = df_new.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['x'],data_with_clusters['y'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[ ]:




