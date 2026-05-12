# ============================================================
# FAKE NEWS DETECTION — Step 7: Results & Analysis
# Side-by-side comparison: BiLSTM vs DistilBERT
# Prerequisite: run 04_evaluation.py first
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ── Load Saved Evaluation Artifacts ──────────────────────────
y_test = np.load('y_test.npy', allow_pickle=True)

bilstm_metrics       = np.load('bilstm_metrics.npy',           allow_pickle=True).item()
bilstm_y_pred        = np.load('bilstm_y_pred.npy',            allow_pickle=True)
bilstm_y_pred_prob   = np.load('bilstm_y_pred_prob.npy',       allow_pickle=True)

distilbert_metrics     = np.load('distilbert_metrics.npy',     allow_pickle=True).item()
distilbert_y_pred      = np.load('distilbert_y_pred.npy',      allow_pickle=True)
distilbert_y_pred_prob = np.load('distilbert_y_pred_prob.npy', allow_pickle=True)

# Use y_test as the ground truth for both (same test split)
y_true = y_test

# ============================================================
# 1 — Metrics Summary Table
# ============================================================

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
bilstm_vals   = [bilstm_metrics['accuracy'], bilstm_metrics['precision'],
                 bilstm_metrics['recall'],    bilstm_metrics['f1'],    bilstm_metrics['roc_auc']]
bert_vals     = [distilbert_metrics['accuracy'], distilbert_metrics['precision'],
                 distilbert_metrics['recall'],    distilbert_metrics['f1'], distilbert_metrics['roc_auc']]

print("=" * 55)
print(f"{'Metric':<14} {'BiLSTM':>12} {'DistilBERT':>12} {'Delta':>10}")
print("=" * 55)
for name, bv, dv in zip(metrics_names, bilstm_vals, bert_vals):
    delta = dv - bv
    sign  = '+' if delta >= 0 else ''
    print(f"{name:<14} {bv*100:>11.2f}% {dv*100:>11.2f}% {sign}{delta*100:>8.2f}%")
print("=" * 55)

# ============================================================
# 2 — Side-by-Side Confusion Matrix Comparison
# ============================================================

cm_bilstm = confusion_matrix(y_true, bilstm_y_pred)
cm_bert   = confusion_matrix(y_true, distilbert_y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, cm, title, cmap in [
    (axes[0], cm_bilstm, 'BiLSTM — Confusion Matrix',    'Blues'),
    (axes[1], cm_bert,   'DistilBERT — Confusion Matrix', 'Oranges')
]:
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Predicted Fake', 'Predicted Real'],
                yticklabels=['Actual Fake', 'Actual Real'],
                linewidths=0.5, linecolor='white', annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: confusion_matrix_comparison.png")

# ============================================================
# 3 — ROC Curve Comparison
# ============================================================

fpr_bilstm, tpr_bilstm, _ = roc_curve(y_true, bilstm_y_pred_prob)
fpr_bert,   tpr_bert,   _ = roc_curve(y_true, distilbert_y_pred_prob)
auc_bilstm                 = roc_auc_score(y_true, bilstm_y_pred_prob)
auc_bert                   = roc_auc_score(y_true, distilbert_y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr_bilstm, tpr_bilstm, color='#2E75B6', linewidth=2.5, label=f'BiLSTM     (AUC = {auc_bilstm:.4f})')
plt.plot(fpr_bert,   tpr_bert,   color='#E07B2E', linewidth=2.5, label=f'DistilBERT (AUC = {auc_bert:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Baseline (AUC = 0.5000)')
plt.fill_between(fpr_bilstm, tpr_bilstm, alpha=0.06, color='#2E75B6')
plt.fill_between(fpr_bert,   tpr_bert,   alpha=0.06, color='#E07B2E')
plt.title('ROC Curve Comparison: BiLSTM vs DistilBERT', fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150)
plt.show()
print("Saved: roc_comparison.png")

# ============================================================
# 4 — Bar Chart: All Metrics Side by Side
# ============================================================

x     = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, [v * 100 for v in bilstm_vals], width, label='BiLSTM',     color='#2E75B6', edgecolor='white')
bars2 = ax.bar(x + width/2, [v * 100 for v in bert_vals],   width, label='DistilBERT', color='#E07B2E', edgecolor='white')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{bar.get_height():.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title('BiLSTM vs DistilBERT — Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_ylim(90, 101)
ax.legend(fontsize=12)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('metrics_comparison_bar.png', dpi=150)
plt.show()
print("Saved: metrics_comparison_bar.png")

# ============================================================
# 5 — Error Analysis: False Positive / Negative Breakdown
# ============================================================

print("\n── Error Analysis ──────────────────────────────────")
for model_name, y_pred, cm in [
    ('BiLSTM',     bilstm_y_pred,     cm_bilstm),
    ('DistilBERT', distilbert_y_pred, cm_bert)
]:
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)
    print(f"\n{model_name}:")
    print(f"  True Negatives  (Fake → Fake): {tn:4d}  ({tn/total*100:.2f}%)")
    print(f"  False Positives (Fake → Real): {fp:4d}  ({fp/total*100:.2f}%)  ← fake called real")
    print(f"  False Negatives (Real → Fake): {fn:4d}  ({fn/total*100:.2f}%)  ← real called fake")
    print(f"  True Positives  (Real → Real): {tp:4d}  ({tp/total*100:.2f}%)")

# ============================================================
# 6 — Training Curve Comparison (if history files exist)
# ============================================================

try:
    bilstm_acc  = np.load('bilstm_history_acc.npy')
    bilstm_vacc = np.load('bilstm_history_valacc.npy')
    bert_acc    = np.load('distilbert_history_acc.npy')
    bert_vacc   = np.load('distilbert_history_valacc.npy')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(range(1, len(bilstm_acc)+1),  bilstm_acc,  label='BiLSTM Train',     color='#2E75B6', linewidth=2)
    axes[0].plot(range(1, len(bilstm_vacc)+1), bilstm_vacc, label='BiLSTM Val',       color='#2E75B6', linewidth=2, linestyle='--')
    axes[0].plot(range(1, len(bert_acc)+1),    bert_acc,    label='DistilBERT Train', color='#E07B2E', linewidth=2)
    axes[0].plot(range(1, len(bert_vacc)+1),   bert_vacc,   label='DistilBERT Val',   color='#E07B2E', linewidth=2, linestyle='--')
    axes[0].set_title('Training Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    bilstm_loss  = np.load('bilstm_history_loss.npy')
    bilstm_vloss = np.load('bilstm_history_valloss.npy')
    bert_loss    = np.load('distilbert_history_loss.npy')
    bert_vloss   = np.load('distilbert_history_valloss.npy')

    axes[1].plot(range(1, len(bilstm_loss)+1),  bilstm_loss,  label='BiLSTM Train',     color='#2E75B6', linewidth=2)
    axes[1].plot(range(1, len(bilstm_vloss)+1), bilstm_vloss, label='BiLSTM Val',       color='#2E75B6', linewidth=2, linestyle='--')
    axes[1].plot(range(1, len(bert_loss)+1),    bert_loss,    label='DistilBERT Train', color='#E07B2E', linewidth=2)
    axes[1].plot(range(1, len(bert_vloss)+1),   bert_vloss,   label='DistilBERT Val',   color='#E07B2E', linewidth=2, linestyle='--')
    axes[1].set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('BiLSTM vs DistilBERT — Training History', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('training_history_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: training_history_comparison.png")

except FileNotFoundError:
    print("Training history files not found — skipping training curve comparison.")

print("\nAll analysis complete.")
print("Output files: confusion_matrix_comparison.png, roc_comparison.png,")
print("              metrics_comparison_bar.png, training_history_comparison.png")
