#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:39:43 2022

@author: bangufe
"""

import pandas as pd
import os

filename = "label_accounting.csv"
folder = os.path.join("/","Data", "CREMEv2 Result", "20221126", "label_accounting")
model_base_folder = os.path.join(os.getcwd(), 'model_' + filename.split(".")[0].split("_")[1])

if os.path.exists(model_base_folder):
    pass
else:
    os.mkdir(model_base_folder)

#%%

df = pd.read_csv(os.path.join(folder, filename))
X = df.drop(columns=['Label'])
X = X.values
y = df['Label'].values
print(y)
y = y.reshape(-1)

#%%

# linear models
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
# non-linear models
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# ensemble models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pickle

# balance data
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

core = -1 # your computer's maximum core numbers, for improving the training speed


# classes = np.unique(y_train)
# sw = compute_sample_weight(class_weight='balance', y=y_train)
models = {}
# linear models
models['Logistic_Regression'] = LogisticRegression(n_jobs=core)
models['SGD'] = SGDClassifier(n_jobs=core)
models['Passive_Aggressive'] = PassiveAggressiveClassifier(n_jobs=core)
# non-linear models
models['Decision_Tree'] = DecisionTreeClassifier()
models['Extra_Tree'] = ExtraTreeClassifier()
models['Gaussian_NB'] = GaussianNB()
models['SVC'] = SVC(kernel='rbf', gamma='auto')
models['KNeighbors'] = KNeighborsClassifier(n_jobs=core)
# ensemble models
# models['XGB'] = XGBClassifier(n_jobs=core)
models['Random_Forest'] = RandomForestClassifier(n_jobs=core)
models['Ada_Boost'] = AdaBoostClassifier()
models['Bagging'] = BaggingClassifier(n_jobs=core)
models['Extra_Trees'] = ExtraTreesClassifier(n_jobs=core)
models['Gradient_Boosting'] = GradientBoostingClassifier()

#%%
import os
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# training and testing
evaluation = {}
for name in models:
    evaluation[name] = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
for i in range(5):
    # split the dataset into training set and testing set
    # specify a number for random_state if you want to make the splitting result all the same
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i*100+29)
    for name, model in models.items():
        model_filename = os.path.join(model_base_folder, name + '_' + str(i+1))
        if os.path.exists(model_filename): # load the model from disk
            model = pickle.load(open(model_filename, 'rb'))
        else:
            print('Training model {} - count {}'.format(name, i+1))
            model.fit(X_train, y_train) # save the model to disk
            pickle.dump(model, open(model_filename, 'wb'))
        y_hat = model.predict(X_test)
        # evaluation
        # balance_accuracy_score
        evaluation[name]['accuracy'].append(accuracy_score(y_test, y_hat))
        evaluation[name]['precision'].append(precision_score(y_test, y_hat, average='weighted'))
        evaluation[name]['recall'].append(recall_score(y_test, y_hat, average='weighted'))
        evaluation[name]['f1_score'].append(f1_score(y_test, y_hat, average='weighted'))
