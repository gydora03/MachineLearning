# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:55:14 2020

@author: Gyorgy Dora
"""


import numpy as np;
import pandas as pd
from urllib.request import urlopen;
import matplotlib.pyplot as plt;
from sklearn import linear_model, model_selection, neural_network, metrics, datasets, svm;
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc, roc_auc_score, plot_roc_curve;


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

describeStat = dataFrame.describe()    # descriptive statistics
dataFrame.plot()
plt.show()


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


# Particionálás tanító és teszt adatállományra
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, shuffle = True, test_size=0.3, random_state=2020);


#--------------------------------------------------------------------------------------------------


# LINEÁRIS REGRESSZIÓ

# Tanítás scikit-learn logistic regression osztállyal
linreg = linear_model.LinearRegression();
linreg.fit(X_train,y_train);    # fitting the model to data
intercept_linreg = linreg.intercept_;   # intecept (constant) parameter
coef_linreg = linreg.coef_;     # regression coefficients (weights)
score_train_linreg = linreg.score(X_train,y_train);
score_test_linreg = linreg.score(X_test,y_test);    #  R-square for goodness of fit
y_train_pred_linreg = linreg.predict(X_train)
y_test_pred_linreg = linreg.predict(X_test);    # prediction by sklearn

# A célváltozó (target) és előrejelzett értékének az összehasonlítása ábrán
plt.figure(2);
plt.title('Website Phishing Linear regression');
plt.xlabel('True disease progression');
plt.ylabel('Predicted disease progression');
plt.scatter(y_test, y_test_pred_linreg, color="blue");
plt.plot([-2,2],[-2,2],color='red');
plt.show(); 

# Prediction for whole dataset
linreg_pred = linreg.predict(X);  # prediction by sklearn
linreg_pred1 = intercept_linreg*np.ones((rows))+np.dot(X, coef_linreg);  # prediction by numpy
error = y - linreg_pred1;  # error of prediction
centered_target = y - y.mean(); 
linreg_score = linreg.score(X, y);  # computing R-square by sklearn
linreg_score1 = 1-np.dot(error,error)/np.dot(centered_target, centered_target); # computing R-square by numpy


#--------------------------------------------------------------------------------------------------


# LOGISZTIKUS REGRESSZIÓ

# Tanítás scikit-learn logistic regression osztállyal
logreg = linear_model.LogisticRegression(max_iter=1000000)
logreg.fit(X_train, y_train)
intercept_logreg = logreg.intercept_[0];
coef_logreg = logreg.coef_[0,:];
score_train_logreg = logreg.score(X_train, y_train)
score_test_logreg = logreg.score(X_test, y_test) 
y_train_pred_logreg = logreg.predict(X_train)
y_test_pred_logreg = logreg.predict(X_test)
p_pred_prob_logreg = logreg.predict_proba(X_test)

# A célváltozó (target) és előrejelzett értékének az összehasonlítása ábrán
plt.figure(3);
plt.title('Website Phishing Logistic regression');
plt.xlabel('True disease progression');
plt.ylabel('Predicted disease progression');
plt.scatter(y_test, y_test_pred_logreg, color="blue");
plt.plot([-2,2],[-2,2],color='red');
plt.show(); 

# Tanítás tévesztési (confusion) mátrix segítségével
cm_logreg_train = metrics.confusion_matrix(y_train, y_train_pred_logreg)    # train confusion matrix
cm_logreg_test = metrics.confusion_matrix(y_test, y_test_pred_logreg)    # test confusion matrix

# A nem normalizált tévesztési (confusion) matrix kirajzoltatása train és test esetén is
plot_confusion_matrix(logreg, X_train, y_train, display_labels=target_names);
plt.title('Confusion matrix for train dataset (logistic regression)');
plot_confusion_matrix(logreg, X_test, y_test, display_labels=target_names);
plt.title('Confusion matrix for test dataset (logistic regression)');



#--------------------------------------------------------------------------------------------------


# PERCEPTRON

# Tanítás scikit-learn perceptron osztállyal
perceptron = linear_model.Perceptron(max_iter=1000000);
perceptron.fit(X_train, y_train);
intercept_perceptron = perceptron.intercept_[0];
coef_perceptron = perceptron.coef_[0,0];
score_train_perceptron = perceptron.score(X_train, y_train);
score_test_perceptron = perceptron.score(X_test, y_test);
y_train_pred_perceptron = perceptron.predict(X_train);
y_test_pred_perceptron = perceptron.predict(X_test);

# A célváltozó (target) és előrejelzett értékének az összehasonlítása ábrán
plt.figure(6);
plt.title('Website Phishing Perceptron');
plt.xlabel('True disease progression');
plt.ylabel('Predicted disease progression');
plt.scatter(y_test, y_test_pred_perceptron, color="blue");
plt.plot([-2,2],[-2,2],color='red');
plt.show(); 

# Tanítás tévesztési (confusion) mátrix segítségével
cm_perceptron_train = metrics.confusion_matrix(y_train, y_train_pred_perceptron);    # train confusion matrix
cm_perceptron_test = metrics.confusion_matrix(y_test, y_test_pred_perceptron);    # test confusion matrix

# A tévesztési (confusion) matrix kirajzoltatása train és test esetén is
plt.title('Confusion matrix for train dataset (Perceptron)');
plot_confusion_matrix(perceptron, X_train, y_train, display_labels=target_names);
plt.title('Confusion matrix for test dataset (Perceptron)');
plot_confusion_matrix(perceptron, X_test, y_test, display_labels=target_names);


#--------------------------------------------------------------------------------------------------


# NEURÁLIS HÁLÓ

# Tanítás scikit-learn neural network osztállyal
neural = neural_network.MLPClassifier(hidden_layer_sizes=(3),activation='logistic',solver='lbfgs', max_iter=1000000);
neural.fit(X_train, y_train);
intercept_neural = neural.intercepts_[0][0];
coef_neural = neural.coefs_[0][0,0];
score_train_neural = neural.score(X_train, y_train);
score_test_neural = neural.score(X_test, y_test);
y_train_pred_neural = neural.predict(X_train);
y_test_pred_neural = neural.predict(X_test);
p_pred_prob_neural = neural.predict_proba(X_test);

# A célváltozó (target) és előrejelzett értékének az összehasonlítása ábrán
plt.figure(9);
plt.title('Website Phishing Neural network');
plt.xlabel('True disease progression');
plt.ylabel('Predicted disease progression');
plt.scatter(y_test, y_test_pred_neural, color="blue");
plt.plot([-2,2],[-2,2],color='red');
plt.show(); 

# Tanítás tévesztési (confusion) mátrix segítségével
cm_neural_train = metrics.confusion_matrix(y_train, y_train_pred_neural);    # train confusion matrix
cm_neural_test = metrics.confusion_matrix(y_test, y_test_pred_neural);    # test confusion matrix

# A tévesztési (confusion) matrix kirajzoltatása train és test esetén is
plt.title('Confusion matrix for train dataset (Neural network)');
plot_confusion_matrix(neural, X_train, y_train, display_labels=target_names);
plt.title('Confusion matrix for test dataset (Neural network)');
plot_confusion_matrix(neural, X_test, y_test, display_labels=target_names);


#--------------------------------------------------------------------------------------------------


# # # Plotting ROC curve
# plot_roc_curve(logreg, X_test, y_test);
# plot_roc_curve(neural, X_test, y_test);

# fpr_logreg, tpr_logreg, _ = roc_curve(y_test, p_pred_prob_logreg[:,1]);
# roc_auc_logreg = auc(fpr_logreg, tpr_logreg);

# fpr_neural, tpr_neural, _ = roc_curve(y_test, p_pred_prob_neural[:,1]);
# roc_auc_neural = auc(fpr_neural, tpr_neural);

# plt.figure(7);
# lw = 2;
# plt.plot(fpr_logreg, tpr_logreg, color='red',
#          lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
# plt.plot(fpr_neural, tpr_neural, color='blue',
#          lw=lw, label='Neural (AUC = %0.2f)' % roc_auc_neural);
# plt.plot([-1, 1], [-1, 1], color='black', lw=lw, linestyle='--');
# plt.xlim([-1.0, 1.0]);
# plt.ylim([-1.0, 1.0]);
# plt.xlabel('False Positive Rate');
# plt.ylabel('True Positive Rate');
# plt.title('Receiver operating characteristic curve');
# plt.legend(loc="lower right");
# plt.show();


X, y = datasets.make_classification(random_state=0)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
clf = svm.SVC(random_state=0)
clf.fit(X_train, y_train)
metrics.plot_roc_curve(clf, X_test, y_test)
plt.show()  


#--------------------------------------------------------------------------------------------------
