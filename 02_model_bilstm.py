# ============================================================
# FAKE NEWS DETECTION — Step 3: BiLSTM Model Training
# Prerequisite: run 01_data_preprocessing.py first
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM,
    Dense, Dropout, SpatialDropout1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version: {tf.__version__}")

# ── Load Preprocessed Data ───────────────────────────────────
X_train = np.load('X_train_pad.npy')
X_val   = np.load('X_val_pad.npy')
X_test  = np.load('X_test_pad.npy')
y_train = np.load('y_train.npy')
y_val   = np.load('y_val.npy')
y_test  = np.load('y_test.npy')
embedding_matrix = np.load('embedding_matrix.npy')

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
print(f"Embedding matrix: {embedding_matrix.shape}")

# ── Hyperparameters ──────────────────────────────────────────
MAX_LEN      = 256
EMBED_DIM    = 100
VOCAB_SIZE   = embedding_matrix.shape[0]   # 30001

LSTM_UNITS   = 128
DENSE_UNITS  = 64
DROPOUT      = 0.4
SPATIAL_DROP = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE    = 64
EPOCHS        = 10

print(f"\nHyperparameters:")
print(f"  LSTM_UNITS={LSTM_UNITS}, DENSE_UNITS={DENSE_UNITS}")
print(f"  DROPOUT={DROPOUT}, SPATIAL_DROP={SPATIAL_DROP}")
print(f"  BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LEARNING_RATE}")

# ── Build Model ──────────────────────────────────────────────
model = Sequential([
    # GloVe embedding — frozen
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBED_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False
    ),
    SpatialDropout1D(SPATIAL_DROP),
    Bidirectional(LSTM(
        LSTM_UNITS,
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.1
    )),
    Dropout(DROPOUT),
    Dense(DENSE_UNITS, activation='relu'),
    Dropout(DROPOUT),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Callbacks ────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath='bilstm_best.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# ── Training ─────────────────────────────────────────────────
print(f"\nStarting training...")
print(f"Train samples: {len(X_train):,} | Val samples: {len(X_val):,}\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print(f"\nTraining complete.")
print(f"Best val_loss at epoch: {np.argmin(history.history['val_loss']) + 1}")
print(f"Best val_accuracy:      {max(history.history['val_accuracy']):.4f}")

# ── Quick Validation Check ───────────────────────────────────
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Loss:     {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}  ({val_acc*100:.2f}%)")

# ── Plot Training Curves ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='#2E75B6', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='#E05C5C', linewidth=2)
axes[0].set_title('BiLSTM — Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train Loss', color='#2E75B6', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='#E05C5C', linewidth=2)
axes[1].set_title('BiLSTM — Loss', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Binary Cross-Entropy Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('BiLSTM Training History', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('bilstm_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: bilstm_training_curves.png")

# ── Save History ─────────────────────────────────────────────
np.save('bilstm_history_acc.npy',     np.array(history.history['accuracy']))
np.save('bilstm_history_valacc.npy',  np.array(history.history['val_accuracy']))
np.save('bilstm_history_loss.npy',    np.array(history.history['loss']))
np.save('bilstm_history_valloss.npy', np.array(history.history['val_loss']))

print("Saved training history and model weights (bilstm_best.keras).")
print("BiLSTM training complete. Proceed to 03_model_distilbert.py")
