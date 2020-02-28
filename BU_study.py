# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 08:14:14 2019

@author: Andy
"""

import os
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score

os.chdir('C:\\Users\\Andy\\Google Drive\\AAB\\BuruliUlcerProject\\Data Fusion Files')
BU_Study = pd.read_csv('Fused Main Study.csv', encoding='ISO-8859-1')
samples = BU_Study.loc[:, 'Unnamed: 0']
BU_Study = BU_Study.drop('Unnamed: 0', axis=1)
y = BU_Study.loc[:, 'BU status_fame'].values
features = BU_Study.columns.to_list()
X = BU_Study.values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.333)

param_dist = dict(n_estimators=(500, 600, 700, 800), max_depth=[np.random.randint(3, 9)],
                  bootstrap=(True, False), criterion=('entropy', 'gini'))
rand = RandomizedSearchCV(ExtraTreesClassifier(), verbose=10, n_iter= 10, cv=LeaveOneOut(),
                   param_distributions= param_dist, n_jobs= -1)
#rand.fit(X_train, Y_train)
extra = ExtraTreesClassifier(n_estimators=800, criterion='gini', bootstrap='False', max_depth=4)
extra.fit(X_train, Y_train)
print('ExtraTrees Accuracy is :', accuracy_score(Y_test, extra.predict(X_test)))
#all = cross_val_score(extra, X_train, Y_train, cv=LeaveOneOut())
#print(all.mean())

sfs = SequentialFeatureSelector(extra, k_features='parsimonious', forward=False, cv=5, verbose=1, n_jobs=5, scoring='accuracy')
sfs.fit(X_train, Y_train, custom_feature_names=features)
feature_model= pd.DataFrame.from_dict(sfs.get_metric_dict()).T
print(sfs.get_metric_dict())
print(feature_model)