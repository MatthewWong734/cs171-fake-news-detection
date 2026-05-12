# ============================================================
# FAKE NEWS DETECTION — Step 4 & 6: Model Evaluation
# Evaluates both BiLSTM and DistilBERT on the test set
# Prerequisite: run 02_model_bilstm.py and 03_model_distilbert.py first
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ── Device ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load Test Data ───────────────────────────────────────────
X_test_pad = np.load('X_test_pad.npy')          # for BiLSTM (padded sequences)
X_test_raw = np.load('X_test.npy', allow_pickle=True)  # for DistilBERT (raw text)
y_test     = np.load('y_test.npy', allow_pickle=True)

print(f"Test set: {len(y_test)} samples | Fake: {(y_test==0).sum()} | Real: {(y_test==1).sum()}")

# ============================================================
# PART A — BiLSTM Evaluation
# ============================================================

print("\n" + "=" * 50)
print("PART A: BiLSTM Evaluation")
print("=" * 50)

bilstm_model = load_model('bilstm_best.keras')
print("BiLSTM model loaded.")

y_pred_prob_bilstm = bilstm_model.predict(X_test_pad, batch_size=64, verbose=1).flatten()
y_pred_bilstm      = (y_pred_prob_bilstm >= 0.5).astype(int)

accuracy_bilstm  = accuracy_score(y_test, y_pred_bilstm)
precision_bilstm = precision_score(y_test, y_pred_bilstm)
recall_bilstm    = recall_score(y_test, y_pred_bilstm)
f1_bilstm        = f1_score(y_test, y_pred_bilstm)
roc_auc_bilstm   = roc_auc_score(y_test, y_pred_prob_bilstm)

print(f"\nBiLSTM TEST SET RESULTS")
print(f"  Accuracy:  {accuracy_bilstm:.4f}  ({accuracy_bilstm*100:.2f}%)")
print(f"  Precision: {precision_bilstm:.4f}  ({precision_bilstm*100:.2f}%)")
print(f"  Recall:    {recall_bilstm:.4f}  ({recall_bilstm*100:.2f}%)")
print(f"  F1-Score:  {f1_bilstm:.4f}  ({f1_bilstm*100:.2f}%)")
print(f"  ROC-AUC:   {roc_auc_bilstm:.4f}")
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred_bilstm, target_names=['Fake (0)', 'Real (1)']))

cm_bilstm = confusion_matrix(y_test, y_pred_bilstm)
tn, fp, fn, tp = cm_bilstm.ravel()
print(f"True Negatives  (Fake → Fake): {tn}")
print(f"False Positives (Fake → Real): {fp}  ← fake news called real")
print(f"False Negatives (Real → Fake): {fn}  ← real news called fake")
print(f"True Positives  (Real → Real): {tp}")

# Confusion matrix plot
plt.figure(figsize=(7, 5))
sns.heatmap(cm_bilstm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Fake', 'Predicted Real'],
            yticklabels=['Actual Fake', 'Actual Real'],
            linewidths=0.5, linecolor='white', annot_kws={"size": 14, "weight": "bold"})
plt.title('BiLSTM — Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
plt.ylabel('Actual Label', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig('bilstm_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: bilstm_confusion_matrix.png")

# Confidence distribution
plt.figure(figsize=(10, 4))
plt.hist(y_pred_prob_bilstm[y_test == 0], bins=50, alpha=0.6, color='#E05C5C', label='Actual Fake (0)', edgecolor='white')
plt.hist(y_pred_prob_bilstm[y_test == 1], bins=50, alpha=0.6, color='#2E75B6', label='Actual Real (1)', edgecolor='white')
plt.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')
plt.title('BiLSTM — Prediction Confidence Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Probability (Real)')
plt.ylabel('Number of Articles')
plt.legend()
plt.tight_layout()
plt.savefig('bilstm_confidence_dist.png', dpi=150)
plt.show()
print("Saved: bilstm_confidence_dist.png")

# ROC curve
fpr_bilstm, tpr_bilstm, _ = roc_curve(y_test, y_pred_prob_bilstm)
plt.figure(figsize=(7, 5))
plt.plot(fpr_bilstm, tpr_bilstm, color='#2E75B6', linewidth=2.5, label=f'BiLSTM (AUC = {roc_auc_bilstm:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Baseline (AUC = 0.5)')
plt.fill_between(fpr_bilstm, tpr_bilstm, alpha=0.08, color='#2E75B6')
plt.title('BiLSTM — ROC Curve (Test Set)', fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bilstm_roc_curve.png', dpi=150)
plt.show()
print("Saved: bilstm_roc_curve.png")

# Save BiLSTM outputs
bilstm_metrics = {'accuracy': accuracy_bilstm, 'precision': precision_bilstm,
                  'recall': recall_bilstm, 'f1': f1_bilstm, 'roc_auc': roc_auc_bilstm}
np.save('bilstm_metrics.npy',       bilstm_metrics)
np.save('bilstm_y_pred.npy',        y_pred_bilstm)
np.save('bilstm_y_pred_prob.npy',   y_pred_prob_bilstm)
print("BiLSTM evaluation artifacts saved.")

# ============================================================
# PART B — DistilBERT Evaluation
# ============================================================

print("\n" + "=" * 50)
print("PART B: DistilBERT Evaluation")
print("=" * 50)

MAX_LEN    = 256
BATCH_SIZE = 16

class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert_best')
model_bert     = DistilBertForSequenceClassification.from_pretrained('distilbert_best')
model_bert     = model_bert.to(device)
model_bert.eval()
print("DistilBERT best model loaded.")

test_dataset = FakeNewsDataset(X_test_raw, y_test, bert_tokenizer, MAX_LEN)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)
        outputs        = model_bert(input_ids=input_ids, attention_mask=attention_mask)
        probs          = torch.softmax(outputs.logits, dim=1)[:, 1]
        preds          = outputs.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

y_pred_bert      = np.array(all_preds)
y_pred_prob_bert = np.array(all_probs)
y_true           = np.array(all_labels)

accuracy_bert  = accuracy_score(y_true, y_pred_bert)
precision_bert = precision_score(y_true, y_pred_bert)
recall_bert    = recall_score(y_true, y_pred_bert)
f1_bert        = f1_score(y_true, y_pred_bert)
roc_auc_bert   = roc_auc_score(y_true, y_pred_prob_bert)

print(f"\nDistilBERT TEST SET RESULTS")
print(f"  Accuracy:  {accuracy_bert:.4f}  ({accuracy_bert*100:.2f}%)")
print(f"  Precision: {precision_bert:.4f}  ({precision_bert*100:.2f}%)")
print(f"  Recall:    {recall_bert:.4f}  ({recall_bert*100:.2f}%)")
print(f"  F1-Score:  {f1_bert:.4f}  ({f1_bert*100:.2f}%)")
print(f"  ROC-AUC:   {roc_auc_bert:.4f}")
print("\nFull Classification Report:")
print(classification_report(y_true, y_pred_bert, target_names=['Fake (0)', 'Real (1)']))

cm_bert = confusion_matrix(y_true, y_pred_bert)
tn, fp, fn, tp = cm_bert.ravel()
print(f"True Negatives  (Fake → Fake): {tn}")
print(f"False Positives (Fake → Real): {fp}  <- fake news called real")
print(f"False Negatives (Real → Fake): {fn}  <- real news called fake")
print(f"True Positives  (Real → Real): {tp}")

# Confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Predicted Fake', 'Predicted Real'],
            yticklabels=['Actual Fake', 'Actual Real'],
            linewidths=0.5, linecolor='white', annot_kws={"size": 14, "weight": "bold"})
plt.title('DistilBERT — Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
plt.ylabel('Actual Label', fontsize=11)
plt.xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig('distilbert_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: distilbert_confusion_matrix.png")

# Save DistilBERT outputs
np.save('distilbert_y_pred.npy',      y_pred_bert)
np.save('distilbert_y_pred_prob.npy', y_pred_prob_bert)
np.save('distilbert_metrics.npy', {
    'accuracy': accuracy_bert, 'precision': precision_bert,
    'recall': recall_bert, 'f1': f1_bert, 'roc_auc': roc_auc_bert
})
print("DistilBERT evaluation artifacts saved.")
print("\nEvaluation complete. Proceed to 05_results_analysis.py")
