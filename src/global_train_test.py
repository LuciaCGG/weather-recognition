import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)

from tools.feature_extraction import *
from tools.training_tools import *

# Dataset path
path_split = "..\\data\\split"


#####################
# Feature extraction
#####################

# Feature extraction option
option = 0
t1=time.time()
set_type = "train"
y_train, X_train, train_names = global_feature_extraction(path_split, set_type, option)
set_type = "test"
y_test, X_test, test_names = global_feature_extraction(path_split, set_type, option)
t2=time.time()
print(f"[INFO] Feature extraction time: {t2-t1:.2f}s")


###########################
# Dimensionality reduce
###########################
pca_reduction=False
tsne_reduction=False
X=np.concatenate((X_train, X_test))
if pca_reduction:
    pca = PCA(n_components=300)
    pca_all = pca.fit_transform(X)
    X_train = pca_all[:len(X_train)]
    X_test = pca_all[len(X_train):]
if tsne_reduction:
    tsne = TSNE(n_components=3, random_state=0)
    tsne_all = tsne.fit_transform(X)
    X_train = tsne_all[:len(X_train)]
    X_test = tsne_all[len(X_train):]  


###################
# Model comparison
###################
results, names=compare_ml_models(path_split,X_train, y_train)


###########################
# Model fitting
###########################

# Parameters:
num_trees=100
seed=9

# Random Forests model
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# Fitting model
clf.fit(X_train, y_train)


###################
# Model prediction
###################
y_pred = clf.predict(X_test)


###################
# Results
###################

# Calculate the different measurements comparing the real labels of the test set with the predicted ones

# Mean accuracy
accuracy = accuracy_score(y_test, y_pred)
labels_translation = {0:"Fog", 1:"Rain", 2:"Sand", 3:"Snow"}
print("RESULTS RF:")
print(f"Accuracy: {accuracy:.4f}")
for i, label in labels_translation.items():
    accuracy_label = accuracy_score(y_test[y_test==[i]], y_pred[y_test==[i]])
    print(f"Accuracy {label}: {accuracy_label:.4f}")

# Precision
precision = precision_score(y_test, y_pred, average='macro')
print(f"Precision: {precision:.4f}")

# Recall
recall = recall_score(y_test, y_pred, average='macro')
print(f"Recall: {recall:.4f}")

# F1-score
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1-score: {f1:.4f}")

# Confusion matrix
confusion_matrix=confusion_matrix(y_test, y_pred)
print(confusion_matrix)


###########
# Save CSV
###########
dict_output={"File names": test_names, "Label":[],"Predictions": []}
for real, prediction in zip(y_test, y_pred):
    dict_output["Label"].append(labels_translation[real])
    dict_output["Predictions"].append(labels_translation[prediction])
df_output=pd.DataFrame.from_dict(dict_output)
df_output.to_csv("..\\Documentation\\Prediction.csv", index = False)

