#!/usr/bin/env python3

__author__ = "V.Pazenko"
__version__ = 1.0

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

# Load preprocessed data
df = pd.read_csv(snakemake.input.data)
X_test, y_test = df.iloc[:,:-1], df.iloc[:,-1]

# Load the trained model
model = joblib.load(snakemake.input.model)

y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred, average='micro')
roc_auc = metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) 

# ROC-curve (save png)
roc_disp = metrics.RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.savefig(snakemake.output.roc_curve)
plt.close()

params = model.get_params()
# save results
with open(snakemake.output.results, "w") as f:
    f.write("Model Evaluation Results\n")
    f.write("-------------------------\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC AUC: {roc_auc}\n\n")
    f.write("Model Parameters:\n")
    for param, value in params.items():
        f.write(f"{param}: {value}\n")
