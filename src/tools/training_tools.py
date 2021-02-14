import glob
import os
import warnings

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def compare_ml_models(path_split, train_features, train_labels, num_trees = 100, seed = 9, plotting = False):
    # Parameters
    train_path = os.path.join(path_split,"train")
    test_path = os.path.join(path_split,"test")
    scoring = "accuracy"

    warnings.filterwarnings('ignore')
    
    # Machine learning models
    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))

    # variables to hold the results and names
    results = []
    names   = []

    ###########
    # TRAINING
    ###########
    # 10-fold cross validation
    for name, model in models:
        kfold = KFold(n_splits=10, shuffle= True, random_state=seed)
        cv_results = cross_val_score(model, train_features, train_labels, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = f"{name} -> Mean: {cv_results.mean()}, STD:{cv_results.std()}"
        print(msg)

    # Boxplot algorithm comparison
    if plotting:
        fig = pyplot.figure()
        fig.suptitle('Machine Learning algorithm comparison')
        ax = fig.add_subplot(111)
        pyplot.boxplot(results)
        ax.set_xticklabels(names)
        pyplot.show()

    return results, names
