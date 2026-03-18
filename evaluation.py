# evaluation.py
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
output_dir = "output_files"
graphs_dir = os.path.join(output_dir, "graphs")
os.makedirs(graphs_dir, exist_ok=True)

report_path = os.path.join(output_dir, "evaluation_report.txt")

# =========================
# Load centralized model results
# =========================
with open(os.path.join(output_dir, "centralized_history.pkl"), "rb") as f:
    centralized_history = pickle.load(f)

y_true_c = np.load(os.path.join(output_dir, "y_test_true_central.npy"))
y_pred_c = np.load(os.path.join(output_dir, "y_test_pred_central.npy"))

# =========================
# Load federated model results
# =========================
with open(os.path.join(output_dir, "federated_history.pkl"), "rb") as f:
    federated_history = pickle.load(f)

y_true_f = np.load(os.path.join(output_dir, "y_val_true_final.npy"))
y_pred_f = np.load(os.path.join(output_dir, "y_val_pred_final.npy"))

# =========================
# Compute metrics function
# =========================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1

# Compute metrics for both approaches
acc_c, prec_c, rec_c, f1_c = compute_metrics(y_true_c, y_pred_c)
acc_f, prec_f, rec_f, f1_f = compute_metrics(y_true_f, y_pred_f)

# =========================
# Save textual report
# =========================
with open(report_path, "w") as f:
    f.write("=== Centralized Evaluation ===\n")
    f.write(f"Accuracy:  {acc_c:.4f}\nPrecision: {prec_c:.4f}\nRecall:    {rec_c:.4f}\nF1-score:  {f1_c:.4f}\n\n")
    f.write("=== Federated Evaluation ===\n")
    f.write(f"Accuracy:  {acc_f:.4f}\nPrecision: {prec_f:.4f}\nRecall:    {rec_f:.4f}\nF1-score:  {f1_f:.4f}\n")

print(f"✅ Evaluation complete. Report saved at {report_path}")

# =========================
# Confusion Matrices
# =========================
cm_c = confusion_matrix(y_true_c, y_pred_c)
cm_f = confusion_matrix(y_true_f, y_pred_f)

# Centralized confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_c, annot=True, fmt="d", cmap="Blues")
plt.title("Centralized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(graphs_dir, "centralized_confusion_matrix.png"), dpi=300)
plt.close()

# Federated confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_f, annot=True, fmt="d", cmap="Greens")
plt.title("Federated Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(graphs_dir, "federated_confusion_matrix.png"), dpi=300)
plt.close()

print(f"✅ Confusion matrices saved in {graphs_dir}")
