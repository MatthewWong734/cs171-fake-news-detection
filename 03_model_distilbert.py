# ============================================================
# FAKE NEWS DETECTION — Step 5: DistilBERT Fine-Tuning
# Prerequisite: run 01_data_preprocessing.py first
# Requires GPU (Colab: Runtime > Change runtime type > T4 GPU)
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

# ── Device Check ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected. Go to Runtime > Change runtime type > T4 GPU")

# ── Load Raw Splits ──────────────────────────────────────────
X_train_raw = np.load('X_train.npy', allow_pickle=True)
X_val_raw   = np.load('X_val.npy',   allow_pickle=True)
X_test_raw  = np.load('X_test.npy',  allow_pickle=True)
y_train     = np.load('y_train.npy', allow_pickle=True)
y_val       = np.load('y_val.npy',   allow_pickle=True)
y_test      = np.load('y_test.npy',  allow_pickle=True)

print(f"Train: {X_train_raw.shape} | Val: {X_val_raw.shape} | Test: {X_test_raw.shape}")

# ── Hyperparameters ──────────────────────────────────────────
MAX_LEN       = 256
BATCH_SIZE    = 16      # keep at 16 for free Colab GPU memory
EPOCHS        = 4       # 3-4 epochs is standard for BERT fine-tuning
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1

# ── Tokenizer ────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

sample = tokenizer(str(X_train_raw[0]), max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
print(f"Tokenizer loaded.")
print(f"Sample input_ids shape:      {sample['input_ids'].shape}")
print(f"Decoded tokens (first 20):   {tokenizer.convert_ids_to_tokens(sample['input_ids'][0][:20])}")

# ── Dataset Class ────────────────────────────────────────────
class FakeNewsDataset(Dataset):
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

# ── DataLoaders ──────────────────────────────────────────────
train_dataset = FakeNewsDataset(X_train_raw, y_train, tokenizer, MAX_LEN)
val_dataset   = FakeNewsDataset(X_val_raw,   y_val,   tokenizer, MAX_LEN)
test_dataset  = FakeNewsDataset(X_test_raw,  y_test,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

# ── Model ────────────────────────────────────────────────────
model_bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model_bert = model_bert.to(device)

total_params     = sum(p.numel() for p in model_bert.parameters())
trainable_params = sum(p.numel() for p in model_bert.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ── Optimizer & Scheduler ────────────────────────────────────
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(model_bert.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

print(f"\nTotal training steps: {total_steps} | Warmup steps: {warmup_steps}")

# ── Eval Helper ──────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss    += outputs.loss.item()
            preds          = outputs.logits.argmax(dim=1)
            correct       += (preds == labels).sum().item()
            total         += labels.size(0)
    return total_loss / len(loader), correct / total

# ── Training Loop ────────────────────────────────────────────
history       = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_loss = float('inf')

print(f"\nStarting DistilBERT fine-tuning on {device}...\n")

for epoch in range(EPOCHS):
    model_bert.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model_bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_bert.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loss    += loss.item()
        preds          = outputs.logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    val_loss, val_acc  = evaluate(model_bert, val_loader, device)
    train_loss_avg     = train_loss / len(train_loader)
    train_acc          = train_correct / train_total

    history['train_loss'].append(train_loss_avg)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss_avg:.4f}  Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_bert.save_pretrained('distilbert_best')
        tokenizer.save_pretrained('distilbert_best')
        print(f"  Best model saved (val_loss={val_loss:.4f})\n")

print(f"\nFine-tuning complete.")
print(f"Best val_loss: {best_val_loss:.4f}")
print(f"Best val_acc:  {max(history['val_acc']):.4f}")

# ── Plot Training Curves ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, EPOCHS+1), history['train_acc'], label='Train Accuracy', color='#2E75B6', linewidth=2, marker='o')
axes[0].plot(range(1, EPOCHS+1), history['val_acc'],   label='Val Accuracy',   color='#E05C5C', linewidth=2, marker='o')
axes[0].set_title('DistilBERT — Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticks(range(1, EPOCHS+1))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, EPOCHS+1), history['train_loss'], label='Train Loss', color='#2E75B6', linewidth=2, marker='o')
axes[1].plot(range(1, EPOCHS+1), history['val_loss'],   label='Val Loss',   color='#E05C5C', linewidth=2, marker='o')
axes[1].set_title('DistilBERT — Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Cross-Entropy Loss')
axes[1].set_xticks(range(1, EPOCHS+1))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('DistilBERT Fine-Tuning History', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('distilbert_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: distilbert_training_curves.png")

# ── Save Training History ────────────────────────────────────
np.save('distilbert_history_acc.npy',     np.array(history['train_acc']))
np.save('distilbert_history_valacc.npy',  np.array(history['val_acc']))
np.save('distilbert_history_loss.npy',    np.array(history['train_loss']))
np.save('distilbert_history_valloss.npy', np.array(history['val_loss']))

print("Training history saved. Model saved to: distilbert_best/")
print("DistilBERT training complete. Proceed to 04_evaluation.py")
