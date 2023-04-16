#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:46:23 2022

@author: vasanthdhanagopal
"""
############################ Hirerarcial Clustering ###########################
# A Hierarchical clustering method works via grouping data into a tree of clusters. 
# Hierarchical clustering begins by treating every data point as a separate cluster.

import pandas as pd
# used for data manipualation
import matplotlib.pyplot as plt
# used for graph plotting


EW_data = pd.read_excel("/Users/vasanthdhanagopal/Desktop/360DigiTMG/2.DataScience-Sampath/6.Data Mining-Hierarchical Clustering/Assignment/Dataset_Assignment Clustering/EastWestAirlines.xlsx",'data')
# used to read excel file (second sheet)

EW_data.describe() #tells the mean, max and min values
a = EW_data.info()     # informs the dataypes

EW_data.isnull().sum() # checks the null value and adds it

EW_data = EW_data.drop(['ID#'],axis=1)


# The data is checked for gaussian distribution, but most of them are not, also it is not needed for this analysis
# Normal Q-Q plot
# import scipy.stats as stats
# import pylab
# stats.probplot(EW_data['Balance'], dist="norm", plot=pylab)
# import matplotlib.pyplot as plt
# plt.hist(EW_data['Balance'])
# # Use transformation technique 
# import numpy as np
# # used for numerical calculations
# EW_data_log = np.log(EW_data['Balance'])
# df = EW_data_log.to_frame()
# stats.probplot(df['Balance'], dist="norm", plot=pylab)


# Normalization is good to use when you know that the distribution of your data 
# does not follow a Gaussian distribution

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
EW_data_norm  = scaler.fit_transform(EW_data)
EW_data_norm = pd.DataFrame(EW_data_norm)

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

l = linkage(EW_data_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(l, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.axhline(y=1.75, color='r', linestyle='--')
plt.show()

# Now applying AgglomerativeClustering - bottom-top approach
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(EW_data_norm ) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

EW_data['clust'] = cluster_labels # creating a new column and assigning it to new column 

EW_data = EW_data.iloc[:, [11,0,1,2,3,4,5,6,7,8,9,10]]
EW_data.head()

# Aggregate mean of each cluster
EW_data.iloc[:, 2:].groupby(EW_data.clust).mean()

# creating a csv file 
EW_data.to_excel("EW_air.xlsx", encoding = "utf-8")

import os
os.getcwd()


