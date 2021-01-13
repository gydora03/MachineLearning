# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:38:37 2020

@author: Gyorgy Dora
"""


import numpy as np;
import pandas as pd
from urllib.request import urlopen;
import matplotlib.pyplot as plt;
from sklearn.decomposition import PCA;  
from sklearn.cluster import KMeans;
from sklearn.metrics import davies_bouldin_score;


#--------------------------------------------------------------------------------------------------


url = 'https://raw.githubusercontent.com/gydora03/MachineLearning/main/WebsitePhishing/WebsitePhishing.csv';

# WebsitePhishing adatállomány importolása a Pandas segítségével
dataFrame = pd.read_csv(url)
X = dataFrame.iloc[:,0:9].values;    # input attributes, WebsitePhishing data
y = dataFrame.iloc[:,9].values;    # target attribute, WebsitePhishing target

rows = dataFrame.shape[0];    # number of rows
cols = dataFrame.shape[1];    # number of columns
input_names = dataFrame.columns;
target_names = ['Legitimate','Suspicious', 'Phishy'];

# describeStat = dataFrame.describe()    # descriptive statistics
# dataFrame.plot()
# plt.show()


# # WebsitePhishing adatállomány importolása a Numpy segítségével
# raw_data = urlopen(url);    # opening url
# websitePhishingData = np.loadtxt(raw_data, delimiter=",", skiprows=1);    # load dataset
# X = websitePhishingData[:,0:9];    # input attributes, WebsitePhishing data
# y = websitePhishingData[:,9];    # target attribute, WebsitePhishing target

# del url, urlopen, raw_data;
# rows = websitePhishingData.shape[0];    # number of rows
# cols = websitePhishingData.shape[1];    # number of columns
# input_names = ['SFH', 'popUpWidnow', 'SSLfinal_State', 'Request_URL', 'URL_of_Anchor', 
#                 'web_traffic', 'URL_Length','age_of_domain','having_IP_Address','Result'];
# target_names = ['Legitimate','Suspicious', 'Phishy'];


#--------------------------------------------------------------------------------------------------


n_c = 2; # number of clusters

# A klaszterszám bekérése a felhasználótól
user_input = input('Number of clusters [default:3]: ');
if len(user_input) != 0 :
    n_c = np.int8(user_input);

# Kmeans klaszterezés
kmeans = KMeans(n_clusters=n_c, random_state=2020);     # instance of KMeans class
kmeans.fit(X);      # fitting the model to data
wp_labels = kmeans.labels_;     # cluster labels
wp_centers = kmeans.cluster_centers_;       # centroid of clusters
y_pred = kmeans.predict(X);     # predicting cluster label
sse = kmeans.inertia_;      # sum of squares of error (within sum of squares)
score = kmeans.score(X);      # negative error


#--------------------------------------------------------------------------------------------------


# Davies-Bouldin index a modellillesztés jóságának a meghatározásához
DB = davies_bouldin_score(X, wp_labels);

# Az eredmények kiíratása a konzolra
print(f'Number of cluster: {n_c}');
print(f'Within SSE: {sse}');
print(f'Davies-Bouldin index: {DB}');


#--------------------------------------------------------------------------------------------------


# PCA limitált komponensszámmal
pca = PCA(n_components=2);
pca.fit(X);
wp_pc = pca.transform(X);  #  data coordinates in the PC space
centers_pc = pca.transform(wp_centers);  # the cluster centroids in the PC space

# Az adatállomány megjelenítése 2D-s szines PC térben kalszterezés nélkül 
fig = plt.figure(1);
plt.title('Scatterplot for Website Phishing dataset without clustering');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(wp_pc[:,0], wp_pc[:,1], s=50, c=y, cmap = 'tab10');
plt.show();

# A klaszterezés vizualizációja a főkomponensek térében
fig = plt.figure(2);
plt.title('Clustering of the Website Phishing data after PCA');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(wp_pc[:,0], wp_pc[:,1], s=50, c=wp_labels);  # data
plt.scatter(centers_pc[:,0], centers_pc[:,1], s=200, c='red', marker='X');  # centroids
plt.legend();
plt.show();


#--------------------------------------------------------------------------------------------------


distX = kmeans.transform(X);
dist_center = kmeans.transform(wp_centers);

# A klaszterezés vizualizációja a klaszter-középpontoktól való távolságok terébe
fig = plt.figure(3);
plt.title('Website Phising data in the distance space');
plt.xlabel('Cluster 1');
plt.ylabel('Cluster 2');
plt.scatter(distX[:,0], distX[:,1], s=50, c=wp_labels);  # data
plt.scatter(dist_center[:,0], dist_center[:,1], s=200, c='red', marker='X');  # centroids
plt.legend();
plt.show();


#--------------------------------------------------------------------------------------------------


# Az optimális klaszter szám megkeresése
Max_K = 31;     # maximum cluster number
SSE = np.zeros((Max_K-2));    #  array for sum of squares errors
DB = np.zeros((Max_K-2));   # array for Davies Bouldin indeces
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(X);
    wp_labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(X, wp_labels);

# Az SSE (négyzetösszeg) érték vizualizációja    
fig = plt.figure(4);
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2, Max_K), SSE, color='red')
plt.show();

# A DB (Davies-Bouldin index) vizulizációja
fig = plt.figure(5);
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2, Max_K), DB, color='blue')
plt.show();


#--------------------------------------------------------------------------------------------------

