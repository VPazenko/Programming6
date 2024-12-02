#!/usr/bin/env python3

__author__ = "Gijs Bakker (modified for snakemake by V.Pazenko)"
__version__ = 1.0

import pandas as pd
from sklearn import *
import joblib

# Load preprocessed data
df = pd.read_csv(snakemake.input[0])
X_train, y_train = df.iloc[:,:-1], df.iloc[:,-1]

model = pipeline.make_pipeline(
    feature_selection.SelectFromModel(ensemble.ExtraTreesClassifier(n_estimators=50), max_features=8),
    preprocessing.StandardScaler(),
    ensemble.RandomForestClassifier(),
)

gridsearch = model_selection.GridSearchCV(
    estimator=model,
    param_grid={
        'selectfrommodel__max_features' : [10, 25 ,50],
        'randomforestclassifier__n_estimators': [50, 100, 200, 500],
        'randomforestclassifier__criterion': ['gini', 'entropy'],
        'randomforestclassifier__max_depth': [5, 10, 25, 50],
        'randomforestclassifier__min_samples_split': [2],
        'randomforestclassifier__max_features': ['sqrt',"log2", None]
    },
    cv=5,
    scoring=metrics.make_scorer(metrics.f1_score, average='micro'),
    n_jobs=-1,
)

gridsearch.fit(X_train, y_train)
best_model = gridsearch.best_estimator_

# Save the trained model
joblib.dump(best_model, snakemake.output[0])

