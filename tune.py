## Load data
import csv
import numpy as np
from sklearn.model_selection import train_test_split

## define a function that translat the string into float in the array
def trans_str2int(strlist):
    intlist = []
    for i in range(len(strlist)):
        intlist.append([])
        for j in range(len(strlist[i])):
            intlist[i].append(int(strlist[i][j]))
    return list(intlist)

X_str = []
Y = []
with open('./data/train.csv') as csvfile:
    lines = csv.reader(csvfile)
    for line in lines:
        Y.append(line[0])
        image = np.array(line[1:])
        X_str.append(image)
    del X_str[0]
    del Y[0]

x_test_str = []
with open('./data/test.csv') as csvfile:
    lines = csv.reader(csvfile)
    for line in lines:
        image = np.array(line)
        x_test_str.append(image)
    del x_test_str[0]

X = trans_str2int(X_str)
x_test = trans_str2int(x_test_str)
y_test = np.zeros(len(x_test))

## normalized the pixels
x_test = (np.array(x_test)-128.0)/128.0
X = (np.array(X)-128.0)/128.0

## split the train data into training and validation part to check the model
## validation set takes 3% data from the whole training data 
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.03, random_state=1)

## tune the parameters for SVM

from sklearn import svm,  grid_search

parameters = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)

print('Start tuning parameters of kernel')

clf.fit(X, Y)
print(clf.best_params_)
best_kernel = clf.best_params_['kernel']
old_score = best_estimator_.score(X,Y)

if clf.best_params_['kernel'] in ['rbf', 'poly', 'sigmoid']:
    tune_step = {'C': 8, 'gamma': 0.8}
    parameters = {'C':[8, 16, 24], 'gamma':[0.8, 1.6, 2.4]}
else:
    tune_step = {'C': 8}
    parameters = {'C':[8, 16, 24]}

import random

svr = svm.SVC(kernel=best_kernel)
print('Start tuning parameters of C (and gamma)')

while True:
    rand = random.randint(0,99)
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.03, random_state=rand)
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(x_train, y_train)
    new_score = best_estimator_.score(x_valid,y_valid)
    print('Parameters: ', clf.best_params_)
    print('Score: ', new_score)
    if abs(new_score-old_score) > 0.05:
        old_score = new_score
        for i in parameters:
            tune_step[i] = tune_step[i]/2
            parameters[i] = [clf.best_params_[i]-tune_step[i], clf.best_params_[i], clf.best_params_[i]+tune_step[i]]
    else:
        break
print('tuning finished!')