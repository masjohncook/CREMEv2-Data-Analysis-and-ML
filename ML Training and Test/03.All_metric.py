# %%

## 1. Initialization
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import time
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, auc, RocCurveDisplay, roc_curve, average_precision_score, precision_recall_curve
import json
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
import seaborn as sns




folder = os.path.join("/", "Data", "CREMEv2_Result", "20230310", "logs_working", "toTrain")
train_technique = ["label_accounting_train_technique.csv",
                   "relabel_syslog_train_technique.csv",
                   "label_traffic_train_technique.csv"]
train_lifecycle = ["label_accounting_train_lifecycle.csv",
                   "relabel_syslog_train_lifecycle.csv",
                   "label_traffic_train_lifecycle.csv"]
datas = ["accounting", "syslog", "traffic"]

model_folder = os.path.join("/", "Data", "CREMEv2_Result", "20230310", "logs_working", "model")




for data in train_technique:
    if os.path.exists(os.path.join(folder, data)):
        print("Path is exists: ", data)
    else:
        print("Path is not exists: ", data)

for data in train_lifecycle:
    if os.path.exists(os.path.join(folder, data)):
        print("Path is exists: ", data)
    else:
        print("Path is not exists: ", data)
        
        
        
        
# %%

## 2. Model Definition,Parameters Settings, and Evaluation Definition

r_state = 42
core = 8
# model = XGBClassifier(objective='multi:softprob', eval_metric='merror', n_jobs=core)



models = {}
# model_name_technique = []
# model_name_lifecycle = []

### Linear-based
models['Logistic_Regresion'] = LogisticRegression(max_iter=1500, n_jobs=core, verbose=True)

### Tree-based
models['Decision_Tree'] = DecisionTreeClassifier()

# ### SVM-based
# models['SVM'] = SVC(kernel='linear', gamma='auto', verbose=True)

### Naive bayes
models['Naive_Bayes'] = GaussianNB()

### KNN-based
models['KNN'] = KNeighborsClassifier(n_jobs=core)

### ensemble-based
models['XGBoost'] = XGBClassifier(objective='multi:softprob', eval_metric='merror', n_jobs=core, verbosity=2)



evaluation_technique = {}
evaluation_lifecycle = {}


evaluation_roc_technique = {}
evaluation_roc_lifecycle = {}

evaluation_prauc_technique = {}
evaluation_prauc_lifecycle = {}

## accuracy, precision, recall, and F1-score
for data_type in datas:
    evaluation_technique[data_type] = {}
    for name in models:
        evaluation_technique[data_type][name] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

for data_type in datas:
    evaluation_lifecycle[data_type] = {}
    for name in models:
        evaluation_lifecycle[data_type][name] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

# ROC-AUC
for data_type in datas:
    evaluation_roc_technique[data_type] = {}
    for name in models:
        evaluation_roc_technique[data_type][name] = {
            'roc': []
        }

for data_type in datas:
    evaluation_roc_lifecycle[data_type] = {}
    for name in models:
        evaluation_roc_lifecycle[data_type][name] = {
            'roc': []
        }



## PR-AUC
for data_type in datas:
    evaluation_prauc_technique[data_type] = {}
    for name in models:
        evaluation_prauc_technique[data_type][name] = {
            'precision': {},
            'recall': {},
            'average_precision': {}
        }

for data_type in datas:
    evaluation_prauc_lifecycle[data_type] = {}
    for name in models:
        evaluation_prauc_lifecycle[data_type][name] = {
            'precision': {},
            'recall': {},
            'average_precision': {}
        }




print(evaluation_technique)
print("=========================================================================================")
print(evaluation_lifecycle)
print("=========================================================================================")
print("=========================================================================================")
print(evaluation_roc_technique)
print("=========================================================================================")
print(evaluation_roc_lifecycle)
print("=========================================================================================")
print("=========================================================================================")
print(evaluation_prauc_technique)
print("=========================================================================================")
print(evaluation_prauc_lifecycle)




# %%

## 3.1 Training and evaluating (Technique)


##* Training
##* Evaluation
##  * Precision
##  * Recall
##  * F1-score
##  * ROC-AUC
##  * PR-AUC
##* Data Store

i=1
for data in train_technique:
    for data_type in datas:

        remove_extension = data.split('.')
        name_without_ext = remove_extension[0].split('_')

        if data_type == name_without_ext[1]:
            print(f"Processing dataset {i}: {data}")

            df = pd.read_csv(os.path.join(folder, data))


            label_origin = sorted([int(i) for i in df['Label'].unique()])
            le = preprocessing.LabelEncoder()
            le.fit(df['Label'])
            le_origin_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            origin_le_mapping = dict(zip(le.transform(le.classes_), le.classes_))

            X = df.drop(columns=['Label'])
            X = X.to_numpy()
            # X = X.reshape(-1)
            y = df['Label']
            y = y.to_numpy()
            y = y.reshape(-1)
            y = le.transform(y)

            class_label = list(label_origin)



            print(f"Train Test Split {data}")
            X_train_technique, X_test_technique, y_train_technique, y_test_technique = train_test_split(X, y, test_size = 0.2, random_state=r_state)

            print(f"Data balancing of {data}")
            X_train_technique, y_train_technique = SMOTE(n_jobs=-1, random_state=r_state).fit_resample(X_train_technique, y_train_technique)
            j = 1
            for name, model in models.items():
                model_filename = '{}{}_model_{}_{}_{}.pkl'.format(i, j, name, data_type, name_without_ext[-1])
                print(f"{i}{j}. Model {name} --> filename = {model_filename}")
                print("===================================================")

                start_time = time.time()
                if os.path.exists(os.path.join(model_folder, model_filename)):
                    print("Load ", model_filename)
                    model = pickle.load(open(os.path.join(model_folder, model_filename), 'rb'))
                else:
                    print(f"training model {name} on {data}")
                    model.fit(X_train_technique, y_train_technique)
                    print("Dump ", model_filename)
                    pickle.dump(model, open(os.path.join(model_folder, model_filename), 'wb'))
                y_pred_technique = model.predict(X_test_technique)
                y_score_technique = model.predict_proba(X_test_technique)
                label_binarizer_technique = LabelBinarizer().fit(y_train_technique)
                y_onehot_test_technique = label_binarizer_technique.transform(y_test_technique)

                evaluation_technique[data_type][name]['accuracy'].append(accuracy_score(y_test_technique, y_pred_technique))
                evaluation_technique[data_type][name]['precision'].append(precision_score(y_test_technique, y_pred_technique, average='weighted',zero_division=0))
                evaluation_technique[data_type][name]['recall'].append(recall_score(y_test_technique, y_pred_technique, average='weighted', zero_division=0))
                evaluation_technique[data_type][name]['f1_score'].append(f1_score(y_test_technique, y_pred_technique, average='weighted', zero_division=0))
                evaluation_roc_technique[data_type][name]['roc'] = list((y_test_technique, y_onehot_test_technique, y_score_technique))
                end_time = time.time()

                print("Execution Time: {:.2f}\n".format(end_time - start_time))
                j += 1
            else:
                continue
    i += 1
    
    
    
# %%



precision = dict()
recall = dict()
average_precision = dict()



i=1
for data in train_technique:
    for data_type in datas:

        remove_extension = data.split('.')
        name_without_ext = remove_extension[0].split('_')

        if data_type == name_without_ext[1]:
            print(f"Processing dataset {i}: {data}")

            df = pd.read_csv(os.path.join(folder, data))


            label_origin = sorted([int(i) for i in df['Label'].unique()])
            le = preprocessing.LabelEncoder()
            le.fit(df['Label'])
            le_origin_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            origin_le_mapping = dict(zip(le.transform(le.classes_), le.classes_))

            X = df.drop(columns=['Label'])
            X = X.to_numpy()
            # X = X.reshape(-1)
            y = df['Label']
            y = y.to_numpy()
            y = y.reshape(-1)
            y = le.transform(y)

            class_label = list(label_origin)



            print(f"Train Test Split {data}")
            X_train_technique_prauc, X_test_technique_prauc, y_train_technique_prauc, y_test_technique_prauc = train_test_split(X, y, test_size = 0.2, random_state=r_state)

            print(f"Data balancing of {data}")
            X_train_technique_prauc, y_train_technique_prauc = SMOTE(n_jobs=-1, random_state=r_state).fit_resample(X_train_technique_prauc, y_train_technique_prauc)
            j = 1
            for name, model in models.items():
                model_filename = '{}{}_model_prauc_{}_{}_{}.pkl'.format(i, j, name, data_type, name_without_ext[-1])
                print(f"{i}{j}. Model {name} --> filename = {model_filename}")
                print("===================================================")

                start_time = time.time()
                if os.path.exists(os.path.join(model_folder, model_filename)):
                    print("Load ", model_filename)
                    model = pickle.load(open(os.path.join(model_folder, model_filename), 'rb'))
                else:
                    print(f"training model {name} on {data}")
                    model = OneVsRestClassifier(model).fit(X_train_technique_prauc, y_train_technique_prauc)
                    print("Dump ", model_filename)
                    pickle.dump(model, open(os.path.join(model_folder, model_filename), 'wb'))
                y_score_technique_prauc = model.decision_function(X_test_technique_prauc)
                label_binarizer_technique_prauc = LabelBinarizer().fit(y_train_technique_prauc)
                y_onehot_test_technique_prauc = label_binarizer_technique_prauc.transform(y_test_technique_prauc)

                for k in range(y):
                    precision[k], recall[k] = precision_recall_curve(y_test_technique_prauc[:, k], y_score_technique_prauc[:, k])
                    average_precision[k] = average_precision_score(y_test_technique_prauc[:, k], y_score_technique_prauc[:, k])

                evaluation_prauc_technique[data_type][name]['precision'].append(precision)
                evaluation_prauc_technique[data_type][name]['recall'].append(recall)
                evaluation_prauc_technique[data_type][name]['average_precision'].append(average_precision)


                end_time = time.time()

                print("Execution Time: {:.2f}\n".format(end_time - start_time))
                j += 1
            else:
                continue
    i += 1




# %%
print(evaluation_technique)
print("=======================================================================================================")
print(evaluation_roc_technique)
print("=======================================================================================================")
print(evaluation_prauc_technique)




# %%
json_filename = "evaluation_result_technique.json"
with open(os.path.join(model_folder, json_filename), 'w') as json_file:
    json.dump(evaluation_technique, json_file)
  
  
  
    
# %%
json_filename = "evaluation_result_roc_technique.json"
with open(os.path.join(model_folder, json_filename), 'w') as json_file:
    json.dump(evaluation_roc_technique, json_file)
   
   
   
    
# %%
json_filename = "evaluation_result_prauc_technique.json"
with open(os.path.join(model_folder, json_filename), 'w') as json_file:
    json.dump(evaluation_prauc_technique, json_file)
 
 
 
    
# %%

## 3.2. Training and evaluating (Lifecycle)
## * Training
## * Evaluation
##   * Precision
##   * Recall
##   * F1-score
##   * ROC-AUC
##   * PR-AUC
## * Data Store

i=1
for data in train_lifecycle:
    for data_type in datas:

        remove_extension = data.split('.')
        name_without_ext = remove_extension[0].split('_')

        if data_type == name_without_ext[1]:
            print(f"Processing dataset {i}: {data}")

            df = pd.read_csv(os.path.join(folder, data))


            label_origin = sorted([int(i) for i in df['Label_lifecycle'].unique()])
            le = preprocessing.LabelEncoder()
            le.fit(df['Label_lifecycle'])
            le_origin_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            origin_le_mapping = dict(zip(le.transform(le.classes_), le.classes_))

            X = df.drop(columns=['Label_lifecycle'])
            X = X.to_numpy()
            # X = X.reshape(-1)
            y = df['Label_lifecycle']
            y = y.to_numpy()
            y = y.reshape(-1)
            y = le.transform(y)

            class_label = list(label_origin)



            print(f"Train Test Split {data}")
            X_train_lifecycle, X_test_lifecycle, y_train_lifecycle, y_test_lifecycle = train_test_split(X, y, test_size = 0.2, random_state=r_state)

            print(f"Data balancing of {data}")
            X_train_lifecycle, y_train_lifecycle = SMOTE(n_jobs=-1, random_state=r_state).fit_resample(X_train_lifecycle, y_train_lifecycle)
            j = 1
            for name, model in models.items():
                model_filename = '{}{}_model_{}_{}_{}.pkl'.format(i, j, name, data_type, name_without_ext[-1])
                print(f"{i}{j}. Model {name} --> filename = {model_filename}")
                print("===================================================")

                start_time = time.time()
                if os.path.exists(os.path.join(model_folder, model_filename)):
                    print("Load ", model_filename)
                    model = pickle.load(open(os.path.join(model_folder, model_filename), 'rb'))
                else:
                    print(f"training model {name} on {data}")
                    model.fit(X_train_lifecycle, y_train_lifecycle)
                    print("Dump ", model_filename)
                    pickle.dump(model, open(os.path.join(model_folder, model_filename), 'wb'))
                y_pred_lifecycle = model.predict(X_test_lifecycle)
                y_score_lifecycle = model.predict_proba(X_test_lifecycle)
                label_binarizer_lifecycle = LabelBinarizer().fit(y_train_lifecycle)
                y_onehot_test_lifecycle = label_binarizer_lifecycle.transform(y_test_lifecycle)

                evaluation_lifecycle[data_type][name]['accuracy'].append(accuracy_score(y_test_lifecycle, y_pred_lifecycle))
                evaluation_lifecycle[data_type][name]['precision'].append(precision_score(y_test_lifecycle, y_pred_lifecycle, average='weighted',zero_division=0))
                evaluation_lifecycle[data_type][name]['recall'].append(recall_score(y_test_lifecycle, y_pred_lifecycle, average='weighted', zero_division=0))
                evaluation_lifecycle[data_type][name]['f1_score'].append(f1_score(y_test_lifecycle, y_pred_lifecycle, average='weighted', zero_division=0))
                evaluation_roc_lifecycle[data_type][name]['roc'] = list((y_test_lifecycle, y_onehot_test_lifecycle, y_score_lifecycle))
                end_time = time.time()

                print("Execution Time: {:.2f}\n".format(end_time - start_time))
                j += 1
            else:
                continue
    i += 1
    
# %%
precision = dict()
recall = dict()
average_precision = dict()



i=1
for data in train_lifecycle:
    for data_type in datas:

        remove_extension = data.split('.')
        name_without_ext = remove_extension[0].split('_')

        if data_type == name_without_ext[1]:
            print(f"Processing dataset {i}: {data}")

            df = pd.read_csv(os.path.join(folder, data))


            label_origin = sorted([int(i) for i in df['Label_lifecycle'].unique()])
            le = preprocessing.LabelEncoder()
            le.fit(df['Label_lifecycle'])
            le_origin_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            origin_le_mapping = dict(zip(le.transform(le.classes_), le.classes_))

            X = df.drop(columns=['Label_lifecycle'])
            X = X.to_numpy()
            # X = X.reshape(-1)
            y = df['Label_lifecycle']
            y = y.to_numpy()
            y = y.reshape(-1)
            y = le.transform(y)

            class_label = list(label_origin)



            print(f"Train Test Split {data}")
            X_train_lifecycle_prauc, X_test_lifecycle_prauc, y_train_lifecycle_prauc, y_test_lifecycle_prauc = train_test_split(X, y, test_size = 0.2, random_state=r_state)

            print(f"Data balancing of {data}")
            X_train_lifecycle_prauc, y_train_lifecycle_prauc = SMOTE(n_jobs=-1, random_state=r_state).fit_resample(X_train_lifecycle_prauc, y_train_lifecycle_prauc)
            j = 1
            for name, model in models.items():
                model_filename = '{}{}_model_prauc_{}_{}_{}.pkl'.format(i, j, name, data_type, name_without_ext[-1])
                print(f"{i}{j}. Model {name} --> filename = {model_filename}")
                print("===================================================")

                start_time = time.time()
                if os.path.exists(os.path.join(model_folder, model_filename)):
                    print("Load ", model_filename)
                    model = pickle.load(open(os.path.join(model_folder, model_filename), 'rb'))
                else:
                    print(f"training model {name} on {data}")
                    model = OneVsRestClassifier(model).fit(X_train_lifecycle_prauc, Y_train_lifecycle_prauc)
                    print("Dump ", model_filename)
                    pickle.dump(model, open(os.path.join(model_folder, model_filename), 'wb'))
                y_score_lifecycle_prauc = model.decision_function(X_test_lifecycle_prauc)
                label_binarizer_lifecycle_prauc = LabelBinarizer().fit(y_train_lifecycle_prauc)
                y_onehot_test_lifecycle_prauc = label_binarizer_lifecycle_prauc.transform(y_test_lifecycle_prauc)

                for k in range(y):
                    precision[k], recall[k] = precision_recall_curve(y_test_lifecycle_prauc[:, k], y_score_lifecycle_prauc[:, k])
                    average_precision[k] = average_precision_score(y_test_lifecycle_prauc[:, k], y_score_lifecycle_prauc[:, k])

                evaluation_prauc_lifecycle[data_type][name]['precision'].append(precision)
                evaluation_prauc_lifecycle[data_type][name]['recall'].append(recall)
                evaluation_prauc_lifecycle[data_type][name]['average_precision'].append(average_precision)


                end_time = time.time()

                print("Execution Time: {:.2f}\n".format(end_time - start_time))
                j += 1
            else:
                continue
    i += 1
 
 
 
    
# %%
json_filename = "evaluation_result_lifecycle.json"
with open(os.path.join(model_folder, json_filename), 'w') as json_file:
    json.dump(evaluation_lifecycle, json_file)
  
  
  
    
# %%
json_filename = "evaluation_result_lifecycle.json"
with open(os.path.join(model_folder, json_filename), 'w') as json_file:
    json.dump(evaluation_roc_lifecycle, json_file)




# %%
json_filename = "evaluation_result_lifecycle.json"
with open(os.path.join(model_folder, json_filename), 'w') as json_file:
    json.dump(evaluation_prauc_lifecycle, json_file)
    




# %%
## 4.1 Visualization Accuracy, Precision, Recall, and F1-score
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import pickle


def rounded(value, n):
    return math.floor(value * (10 ** n)) / float(10 ** n)

datas = [data for data in evaluation_technique]
models = [name for name in evaluation_technique['accounting']]

print(datas)

result_accounting = {'accuracy': [],
                     'precision': [],
                     'recall': [],
                     'f1_score': []
}

result_syslog = {'accuracy': [],
                 'precision': [],
                 'recall': [],
                 'f1_score': []
}

result_traffic = {'accuracy': [],
                  'precision': [],
                  'recall': [],
                  'f1_score': []
}





# %%
## 4.1.1 Plot Evaluation Accounting
for name in evaluation_technique['accounting']:
    print(name)
    for key, value in evaluation_technique['accounting'][name].items():
        print(key, value)
        result_accounting[key].append(rounded(mean(value), 3))

for name in evaluation_lifecycle['accounting']:
    for key, value in evaluation_lifecycle['accounting'][name].items():
      result_accounting[key].append(rounded(mean(value), 3))


width = 0.2
x = np.arange(len(models))
plt.figure(figsize=(20, 10))
plt.bar(x, result_accounting['accuracy'], width, label='accuracy')
plt.bar(x+width, result_accounting['precision'], width, label='precision')
plt.bar(x+2*width, result_accounting['recall'], width, label='recall')
bar = plt.bar(x+3*width, result_accounting['f1_score'], width, label='f1_score')
plt.bar_label(bar, label_type='edge', fontsize=14)
plt.title('Technique Model Evaluation for Accounting', fontsize=30)
plt.xticks(x+1.5*width, models)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.rcParams.update({
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})
plt.legend(loc='center', fontsize=14, ncol=4, bbox_to_anchor=(0.5,-0.2))
plt.show()





# %%
## 4.1.2 Plot Evaluation Syslog

for name in evaluation_technique['syslog']:
    for key, value in evaluation_technique['syslog'].items():
      result_accounting[key].append(rounded(mean(value)), 3)

for name in evaluation_lifecycle['syslog']:
    for key, value in evaluation_lifecycle['syslog'].items():
      result_accounting[key].append(rounded(mean(value)), 3)


width = 0.2
x = np.arange(len(models))
plt.figure(figsize=(20, 10))
plt.bar(x, result_syslog['accuracy'], width, label='accuracy')
plt.bar(x+width, result_syslog['precision'], width, label='precision')
plt.bar(x+2*width, result_syslog['recall'], width, label='recall')
bar = plt.bar(x+3*width, result_syslog['f1_score'], width, label='f1_score')
plt.bar_label(bar, label_type='edge', fontsize=14)
plt.title('Technique Model Evaluation for Syslog', fontsize=30)
plt.xticks(x+1.5*width, models)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.rcParams.update({
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})
plt.legend(loc='center', fontsize=14, ncol=4, bbox_to_anchor=(0.5,-0.2))
plt.show()

# %%
## 4.1.3 Plot Evaluation Traffic

for name in evaluation_technique['traffic']:
    for key, value in evaluation_technique['traffic'].items():
      result_accounting[key].append(rounded(mean(value)), 3)

for name in evaluation_lifecycle['traffic']:
    for key, value in evaluation_lifecycle['traffic'].items():
      result_accounting[key].append(rounded(mean(value)), 3)


width = 0.2
x = np.arange(len(models))
plt.figure(figsize=(20, 10))
plt.bar(x, result_traffic['accuracy'], width, label='accuracy')
plt.bar(x+width, result_traffic['precision'], width, label='precision')
plt.bar(x+2*width, result_traffic['recall'], width, label='recall')
bar = plt.bar(x+3*width, result_traffic['f1_score'], width, label='f1_score')
plt.bar_label(bar, label_type='edge', fontsize=14)
plt.title('Technique Model Evaluation for Traffic', fontsize=30)
plt.xticks(x+1.5*width, models)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.rcParams.update({
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})
plt.legend(loc='center', fontsize=14, ncol=4, bbox_to_anchor=(0.5,-0.2))
plt.show()




# %%
## 4.2 Visualization Confusion Matrix XGBoost

name = 'Bagging'
model_filename = os.path.join(models_folder, name)
title = "Technique Confusion Matrix of Best Model in {}".format(data_type)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
target_names = label_origin
# model = tf.keras.models.load_model(model_filename)
model = pickle.load(open(model_filename, 'rb'))
y_hat = model.predict(X_test)
# y_hat = np.argmax(y_hat, axis=1)

cm = confusion_matrix(y_test, y_hat)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(15,15))
ax = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
plt.title(title, fontsize=20)
plt.ylabel('Actual', fontsize=14)
plt.xlabel('Predicted', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show(block=False)

# tree.export_graphviz(model)




# %%
## 4.3 Visualization ROC-AUC XGBoost

roc_data_micro = dict()

for data in model_name:
    # stores all informations
    tpr, fpr, roc_auc = dict(), dict(), dict()
    fpr['micro'],  tpr['micro'], _ = roc_curve(evaluation_data[data][1].ravel(), evaluation_data[data][2].ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    roc_data_micro[data] = list((fpr['micro'], tpr['micro'], roc_auc['micro']))

    print("Micro average : {:.2f}".format(roc_auc['micro']))
 
 
 
    
# %%
roc_data_macro = dict()

for data in model_name:
    for i in range(len(class_label)):
        fpr[i], tpr[i], _ = roc_curve(evaluation_data[data][1][:, i], evaluation_data[data][2][:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(len(class_label)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])

    mean_tpr /= len(class_label)

    fpr['macro'] = fpr_grid
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    roc_data_macro[data] = list((fpr['macro'], tpr['macro'], roc_auc['macro']))



    print("Macro average of  : {:.2f}".format(roc_auc['macro']))
    




# %%


fig, ax = plt.subplots(figsize=(10, 10))
legend_data = ["TA ",
               "TS",
               "TT",
               "LA",
               "LS",
               "LT"
               ]


plt.plot(
    roc_data_micro['model_xgboost_accounting_technique.pkl'][0],
    roc_data_micro['model_xgboost_accounting_technique.pkl'][1],
    label="Micro of {}=>AUC={:.2f}".format(legend_data[0], roc_data_micro['model_xgboost_accounting_technique.pkl'][2]),
    color='red',
    linestyle='solid'
)

plt.plot(
    roc_data_macro['model_xgboost_accounting_technique.pkl'][0],
    roc_data_macro['model_xgboost_accounting_technique.pkl'][1],
    label="Macro of {}=>AUC={:.2f}".format(legend_data[0], roc_data_macro['model_xgboost_accounting_technique.pkl'][2]),
    color='red',
    linestyle='dashed'
)

plt.plot(
    roc_data_micro['model_xgboost_syslog_technique.pkl'][0],
    roc_data_micro['model_xgboost_syslog_technique.pkl'][1],
    label="Micro of {}=>AUC={:.2f}".format(legend_data[1], roc_data_micro['model_xgboost_syslog_technique.pkl'][2]),
    color='green',
    linestyle='solid'
)

plt.plot(
    roc_data_macro['model_xgboost_syslog_technique.pkl'][0],
    roc_data_macro['model_xgboost_syslog_technique.pkl'][1],
    label="Macro of {}=>AUC={:.2f}".format(legend_data[1], roc_data_macro['model_xgboost_syslog_technique.pkl'][2]),
    color='green',
    linestyle='dashed'
)

plt.plot(
    roc_data_micro['model_xgboost_traffic_technique.pkl'][0],
    roc_data_micro['model_xgboost_traffic_technique.pkl'][1],
    label="Micro of {}=>AUC={:.2f}".format(legend_data[2], roc_data_micro['model_xgboost_traffic_technique.pkl'][2]),
    color='blue',
    linestyle='solid'
)

plt.plot(
    roc_data_macro['model_xgboost_traffic_technique.pkl'][0],
    roc_data_macro['model_xgboost_traffic_technique.pkl'][1],
    label="Macro of {}=>AUC={:.2f}".format(legend_data[2], roc_data_macro['model_xgboost_traffic_technique.pkl'][2]),
    color='blue',
    linestyle='dashed'
)

plt.plot(
    roc_data_micro['model_xgboost_accounting_lifecycle.pkl'][0],
    roc_data_micro['model_xgboost_accounting_lifecycle.pkl'][1],
    label="Micro of {}=>AUC={:.2f}".format(legend_data[3], roc_data_micro['model_xgboost_accounting_lifecycle.pkl'][2]),
    color='red',
    linestyle='dashdot'
)

plt.plot(
    roc_data_macro['model_xgboost_accounting_lifecycle.pkl'][0],
    roc_data_macro['model_xgboost_accounting_lifecycle.pkl'][1],
    label="Macro of {}=>AUC={:.2f}".format(legend_data[3], roc_data_macro['model_xgboost_accounting_lifecycle.pkl'][2]),
    color='red',
    linestyle='dotted'
)

plt.plot(
    roc_data_micro['model_xgboost_syslog_lifecycle.pkl'][0],
    roc_data_micro['model_xgboost_syslog_lifecycle.pkl'][1],
    label="Micro of {}=>AUC={:.2f}".format(legend_data[4], roc_data_micro['model_xgboost_syslog_lifecycle.pkl'][2]),
    color='green',
    linestyle='dashdot'
)

plt.plot(
    roc_data_macro['model_xgboost_syslog_lifecycle.pkl'][0],
    roc_data_macro['model_xgboost_syslog_lifecycle.pkl'][1],
    label="Macro of {}=>AUC={:.2f}".format(legend_data[4], roc_data_macro['model_xgboost_syslog_lifecycle.pkl'][2]),
    color='green',
    linestyle='dotted'
)

plt.plot(
    roc_data_micro['model_xgboost_traffic_lifecycle.pkl'][0],
    roc_data_micro['model_xgboost_traffic_lifecycle.pkl'][1],
    label="Micro of {}=>AUC={:.2f}".format(legend_data[5], roc_data_micro['model_xgboost_traffic_lifecycle.pkl'][2]),
    color='blue',
    linestyle='dashdot'
)

plt.plot(
    roc_data_macro['model_xgboost_traffic_lifecycle.pkl'][0],
    roc_data_macro['model_xgboost_traffic_lifecycle.pkl'][1],
    label="Macro of {}=>AUC={:.2f}".format(legend_data[5], roc_data_macro['model_xgboost_traffic_lifecycle.pkl'][2]),
    color='blue',
    linestyle='dotted'
)



plt.plot([0, 1], [0, 1], "k--", label="ROC chance level=>AUC=0.5)")
plt.axis("square")
plt.tick_params(axis='both', labelsize=14)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
# plt.title("ROC Curve of Micro and Macro Average for All Dataset in Technique")
plt.legend()
plt.show()





# %%
## 4.4 Visualization PR-AUC XGBoost

from collections import Counter

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
    prevalence_pos_label=Counter(y_test.ravel())[1] / y_test.size,
)
display.plot(plot_chance_level=True)
_ = display.ax_.set_title("Micro-averaged over all classes")





# %%


from itertools import cycle

import matplotlib.pyplot as plt

# setup plot details
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

_, ax = plt.subplots(figsize=(7, 8))

f_scores = np.linspace(0.2, 0.8, num=4)
lines, labels = [], []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

for i, color in zip(range(n_classes), colors):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
        average_precision=average_precision[i],
    )
    display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

# add the legend for the iso-f1 curves
handles, labels = display.ax_.get_legend_handles_labels()
handles.extend([l])
labels.extend(["iso-f1 curves"])
# set the legend and the axes
ax.legend(handles=handles, labels=labels, loc="best")
ax.set_title("Extension of Precision-Recall curve to multi-class")

plt.show()