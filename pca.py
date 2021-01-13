# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:38:37 2020

@author: Gyorgy Dora
"""


import numpy as np;
from urllib.request import urlopen;
import matplotlib.pyplot as plt;
from sklearn import model_selection;
from sklearn.decomposition import PCA;  
import pandas as pd


#--------------------------------------------------------------------------------------------------


url = 'https://raw.githubusercontent.com/gydora03/MachineLearning/main/WebsitePhishing/WebsitePhishing.csv';

# # Import WebsitePhishing dataset with Pandas
# dataFrame = pd.read_csv(url)
# X = dataFrame.iloc[:,0:9].values;    # input attributes, WebsitePhishing data
# y = dataFrame.iloc[:,9].values;    # target attribute, WebsitePhishing target

# rows = dataFrame.shape[0];    # number of rows
# cols = dataFrame.shape[1];    # number of columns
# input_names = dataFrame.columns;
# target_names = ['Legitimate','Suspicious', 'Phishy'];

# describeStat = dataFrame.describe()    # descriptive statistics
# dataFrame.plot()
# plt.show()


# Import WebsitePhishing dataset with Numpy
raw_data = urlopen(url);    # opening url
websitePhishingData = np.loadtxt(raw_data, delimiter=",", skiprows=1);    # load dataset
X = websitePhishingData[:,0:9];    # input attributes, WebsitePhishing data
y = websitePhishingData[:,9];    # target attribute, WebsitePhishing target

del url, urlopen, raw_data;
rows = websitePhishingData.shape[0];    # number of rows
cols = websitePhishingData.shape[1];    # number of columns
input_names = ['SFH', 'popUpWidnow', 'SSLfinal_State', 'Request_URL', 'URL_of_Anchor', 
                'web_traffic', 'URL_Length','age_of_domain','having_IP_Address','Result'];
target_names = ['Legitimate','Suspicious', 'Phishy'];


#--------------------------------------------------------------------------------------------------


# Particionálás tanító és teszt adatállományra
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=2020);


#--------------------------------------------------------------------------------------------------


# PCA DIMENZÓ CSÖKKENTÉS

# A teljes PCA a betanított (train) adatállományra
pca = PCA();
pca.fit(X_train);

# A szórás arányának megjelenítése, amely a fő összetevők fontosságát méri
fig = plt.figure(2);
plt.title('Explained variance ratio plot');
variance_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(variance_ratio))+1;
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos, variance_ratio, align='center', alpha=0.5);
plt.show(); 

# A halmozott variancia arány megjelenítése, amely az első n PC hatását méri
fig = plt.figure(3);
plt.title('Cumulative explained variance ratio plot');
cumulative_variance_ratio = np.cumsum(variance_ratio);
x_pos = np.arange(len(cumulative_variance_ratio))+1;
plt.xlabel('Principal Components');
plt.ylabel('Variance');
plt.bar(x_pos, cumulative_variance_ratio, align='center', alpha=0.5);
plt.show(); 

# A betanított (train) adatállomány megjelenítése 2D-s szines PC térben 
PC_train = pca.transform(X_train);
fig = plt.figure(4);
plt.title('Scatterplot for training Website Phishing dataset');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(PC_train[:,0], PC_train[:,1], s=50, c=y_train, cmap = 'tab10');
plt.show();

# A teszt (test) adatállomány megjelenítése 2D-s PC térben
PC_test = pca.transform(X_test);
fig = plt.figure(5);
plt.title('Scatterplot for test Website Phishing dataset');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(PC_test[:,0], PC_test[:,1], s=50, c=y_test, cmap = 'tab10');
plt.show();


#--------------------------------------------------------------------------------------------------


