"""
Exports everything the website needs into website/data.js
(images as base64 + PCA coordinates + SVM / Random Forest results).

Run after 2_cv_feature_extraction.py:
    python website/build_data.py
"""
import os
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import base64
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(HERE)

data = np.load(os.path.join(BASE_DIR, "cv_features.npz"), allow_pickle=True)
X, y, paths = data["X"], data["y"], data["paths"]

# Re-resolve paths relative to this project (works even if the folder moved)
def local_path(p):
    p = str(p).replace("\\", "/")
    cls = "left_hand" if "/left_hand/" in p else "right_hand"
    return os.path.join(BASE_DIR, "dataset", cls, os.path.basename(p))

paths = [local_path(p) for p in paths]

# ---- Same pipeline as 3_pr_classifier_demo.py ----
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=min(20, X.shape[0] // 2), random_state=42)
X_pca = pca.fit_transform(X_scaled)

idx = np.arange(len(y))
X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
    X_pca, y, idx, test_size=0.25, random_state=42, stratify=y)

svm = SVC(kernel="rbf", C=2.0, probability=True, random_state=42).fit(X_tr, y_tr)
rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)

svm_pred = svm.predict(X_te)
rf_pred = rf.predict(X_te)
svm_prob = svm.predict_proba(X_te)
rf_prob = rf.predict_proba(X_te)

# Out-of-fold probabilities so EVERY image has an honest prediction
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Scaler + PCA are refit inside each fold (same as 3_pr_classifier_demo.py)
n_pc = min(20, X.shape[0] // 2)
cv_svm = cross_val_predict(make_pipeline(StandardScaler(), PCA(n_pc, random_state=42),
                                         SVC(kernel="rbf", C=2.0, probability=True, random_state=42)),
                           X, y, cv=skf, method="predict_proba")
cv_rf = cross_val_predict(make_pipeline(StandardScaler(), PCA(n_pc, random_state=42),
                                        RandomForestClassifier(n_estimators=100, random_state=42)),
                          X, y, cv=skf, method="predict_proba")

cv_svm_pred = cross_val_predict(make_pipeline(StandardScaler(), PCA(n_pc, random_state=42),
                                              SVC(kernel="rbf", C=2.0, random_state=42)), X, y, cv=skf)

test_pos = {int(i): k for k, i in enumerate(i_te)}

samples = []
for i, p in enumerate(paths):
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    s = {
        "name": os.path.basename(p),
        "label": int(y[i]),
        "img": "data:image/png;base64," + b64,
        "pca": [round(float(v), 4) for v in X_pca[i]],
        "split": "test" if i in test_pos else "train",
        "cvSvm": round(float(cv_svm[i, 1]), 4),
        "cvRf": round(float(cv_rf[i, 1]), 4),
        "cvSvmPred": int(cv_svm_pred[i]),
    }
    if i in test_pos:
        k = test_pos[i]
        s["svmProb"] = round(float(svm_prob[k, 1]), 4)
        s["rfProb"] = round(float(rf_prob[k, 1]), 4)
    samples.append(s)

out = {
    "nFeatures": int(X.shape[1]),
    "featureBreakdown": {"colorHist": 192, "activity": 256, "mirror": 128},
    "explainedVariance": [round(float(v), 5) for v in pca.explained_variance_ratio_],
    "svm": {
        "acc": round(accuracy_score(y_te, svm_pred) * 100, 1),
        "cm": confusion_matrix(y_te, svm_pred).tolist(),
        "cvAcc": round(float(np.mean(cv_svm_pred == y)) * 100, 1),
    },
    "rf": {
        "acc": round(accuracy_score(y_te, rf_pred) * 100, 1),
        "cm": confusion_matrix(y_te, rf_pred).tolist(),
        "cvAcc": round(float(np.mean((cv_rf[:, 1] > 0.5) == y)) * 100, 1),
    },
    "samples": samples,
}

with open(os.path.join(HERE, "data.js"), "w", encoding="utf-8") as f:
    f.write("window.BRAIN_DATA = ")
    json.dump(out, f, separators=(",", ":"))
    f.write(";\n")

print(f"SVM test acc {out['svm']['acc']}% | RF test acc {out['rf']['acc']}%")
print(f"SVM 5-fold acc {out['svm']['cvAcc']}% | RF 5-fold acc {out['rf']['cvAcc']}%")
print(f"Wrote website/data.js with {len(samples)} samples")
