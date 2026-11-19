import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # project folder, works anywhere
data = np.load(os.path.join(BASE_DIR, "cv_features.npz"), allow_pickle=True)
X, y, paths = data["X"], data["y"], data["paths"]

print("[3/3] Running Pattern Recognition (PCA + SVM / Random Forest)...")

# 1. Normalization & PCA Dimensionality Reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=min(20, X.shape[0] // 2), random_state=42)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X_pca, y, paths, test_size=0.25, random_state=42, stratify=y
)

# 2. Train Pattern Recognition Classifiers
svm_clf = SVC(kernel="rbf", C=2.0, probability=True, random_state=42)
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
y_prob = svm_clf.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred) * 100

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_clf.predict(X_test)) * 100

print(f"Single 25% split  -> SVM: {acc:.1f}% | Random Forest: {rf_acc:.1f}%")

# 5-fold cross-validation: every image is tested once, so this is the number to trust.
# Scaler and PCA are refit inside each fold, so no information leaks from the test images.
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_pc = min(20, X.shape[0] // 2)
svm_cv = cross_val_score(make_pipeline(StandardScaler(), PCA(n_pc, random_state=42),
                                       SVC(kernel="rbf", C=2.0, random_state=42)), X, y, cv=cv)
rf_cv = cross_val_score(make_pipeline(StandardScaler(), PCA(n_pc, random_state=42),
                                      RandomForestClassifier(n_estimators=100, random_state=42)), X, y, cv=cv)
print(f"5-fold CV         -> SVM: {svm_cv.mean()*100:.1f}% (+/- {svm_cv.std()*100:.1f}) | "
      f"Random Forest: {rf_cv.mean()*100:.1f}% (+/- {rf_cv.std()*100:.1f})")

# 3. Plot Pattern Recognition Summary & Visual Predictions
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])

# PCA 2D Feature Space
ax_pca = fig.add_subplot(gs[0, 0])
for idx, label_name, col in [(0, "Left Hand Intent", "#1f77b4"), (1, "Right Hand Intent", "#d62728")]:
    mask = (y == idx)
    ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label_name, alpha=0.75, c=col, edgecolors="k")
ax_pca.set_title("PCA Feature Space (2D Projection)")
ax_pca.set_xlabel("Principal Component 1")
ax_pca.set_ylabel("Principal Component 2")
ax_pca.legend()
ax_pca.grid(True, alpha=0.3)

# Confusion Matrix
ax_cm = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Left Hand", "Right Hand"])
disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
ax_cm.set_title(f"SVM Confusion Matrix (Acc: {acc:.1f}%)")

# Classifier Comparison Bar Chart
ax_bar = fig.add_subplot(gs[0, 2])
bars = ax_bar.bar(["SVM\nsplit", "SVM\n5-fold", "RF\nsplit", "RF\n5-fold"],
                  [acc, svm_cv.mean() * 100, rf_acc, rf_cv.mean() * 100],
                  color=["#7fa7cf", "#2b5c8f", "#8cc68f", "#388e3c"], width=0.6)
ax_bar.axhline(50, color="#e69500", ls="--", lw=1.5, label="chance (50%)")
ax_bar.legend(loc="upper right")
ax_bar.set_ylim(0, 110)
ax_bar.set_ylabel("Accuracy (%)")
ax_bar.set_title("Single split vs 5-fold cross-validation")
for b in bars:
    ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontweight="bold")

# Bottom row: 3 Sample Brain Image Predictions
label_map = {0: "Move LEFT Hand", 1: "Move RIGHT Hand"}
for i in range(min(3, len(paths_test))):
    ax_img = fig.add_subplot(gs[1, i])
    img = cv2.cvtColor(cv2.imread(paths_test[i]), cv2.COLOR_BGR2RGB)
    ax_img.imshow(img)
    pred_cls = y_pred[i]
    true_cls = y_test[i]
    conf = np.max(y_prob[i]) * 100
    color = "green" if pred_cls == true_cls else "red"
    ax_img.set_title(f"Predicted Intent: {label_map[pred_cls]}\nConfidence: {conf:.1f}% (Actual: {label_map[true_cls]})",
                     color=color, fontweight="bold", fontsize=10)
    ax_img.axis("off")

plt.suptitle("Pattern Recognition & Action Anticipation Results (Pre-Movement EEG Topomaps)", fontsize=14, fontweight="bold")
plt.tight_layout()
out_img = os.path.join(BASE_DIR, "2_pr_results_and_predictions.png")
plt.savefig(out_img, dpi=150)
plt.close(fig)
print(f"Saved full Pattern Recognition report image at: {out_img}")
