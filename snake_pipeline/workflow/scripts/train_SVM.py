#!/usr/bin/env python3

__author__ = "V.Pazenko"
__version__ = 1.0

import pandas as pd
from sklearn import *
import joblib

# Load preprocessed data
df = pd.read_csv(snakemake.input[0])
X_train, y_train = df.iloc[:,:-1], df.iloc[:,-1]

# For roc-curve we need (probability=True)
model = pipeline.make_pipeline(
    feature_selection.SelectFromModel(ensemble.ExtraTreesClassifier(n_estimators=50), max_features=8),
    preprocessing.StandardScaler(),
    svm.SVC(probability=True),
)

gridsearch = model_selection.GridSearchCV(
    estimator=model,
    param_grid={
        'selectfrommodel__max_features': [10, 25, 50],
        'svc__kernel': ['linear', 'rbf', 'poly'],
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto'],
    },
    cv=5,
    scoring=metrics.make_scorer(metrics.f1_score, average='micro'),
    n_jobs=-1,
)

gridsearch.fit(X_train, y_train)
best_model = gridsearch.best_estimator_

# Save the trained model
joblib.dump(best_model, snakemake.output[0])
