import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import os, time
from datetime import date

import itertools
from scipy import sparse
from bisect import bisect_left

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_, 'std':np.std([tree.feature_importances_ for tree in m.estimators_], axis=0)}).sort_values('imp', ascending=False)

def plot_fi(fi, std=True, feature_importance_type=''):
    if std:
        ax = fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False, xerr='std')
    else:
        ax = fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
    
    ax.set_xlabel(f"{feature_importance_type} Feature Importance")
    return ax