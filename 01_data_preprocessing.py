# ============================================================
# FAKE NEWS DETECTION — Step 1 & 2: Data Preprocessing
# WELFake Dataset
# Covers: EDA, cleaning, tokenization, GloVe embeddings
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
import zipfile
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Plot style ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 5)

# ============================================================
# STEP 1 — Load & Inspect Dataset
# ============================================================

# Upload WELFake_Dataset.csv via Colab file upload widget, then set path:
# from google.colab import files
# uploaded = files.upload('WELFake_Dataset.csv')
# uploaded_filename = next(iter(uploaded))
# df = pd.read_csv(uploaded_filename)

# OR if already in Colab environment:
df = pd.read_csv('WELFake_Dataset.csv')

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# ── Dataset Inspection ──────────────────────────────────────
print("=" * 50)
print("COLUMN NAMES:", df.columns.tolist())
print("\nDATA TYPES:")
print(df.dtypes)
print("\nMISSING VALUES:")
print(df.isnull().sum())
print("\nSAMPLE ROWS:")
print(df.sample(5, random_state=42))

# ── Rename & Clean Columns ──────────────────────────────────
df.columns = [col.strip().lower() for col in df.columns]
if 'unnamed: 0' in df.columns:
    df.drop(columns=['unnamed: 0'], inplace=True)
    print("Dropped unnamed index column.")
print("Final columns:", df.columns.tolist())

# ── Drop Nulls ──────────────────────────────────────────────
print(f"\nShape before dropping nulls: {df.shape}")
df.dropna(subset=['title', 'text', 'label'], inplace=True)
print(f"Shape after dropping nulls:  {df.shape}")

print("\nLABEL DISTRIBUTION:")
print(df['label'].value_counts())
print("Label 0 = Fake, Label 1 = Real")

# ── Plot Class Balance ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['label'].value_counts()
ax.bar(['Fake (0)', 'Real (1)'], counts.values, color=['#E05C5C', '#5C9BE0'], edgecolor='white', linewidth=0.8)
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, f'{v:,}', ha='center', fontweight='bold', fontsize=12)
ax.set_title('Class Distribution: Fake vs Real News', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Articles')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()

# ── Text Length Analysis ─────────────────────────────────────
df['title_word_count'] = df['title'].astype(str).apply(lambda x: len(x.split()))
df['text_word_count']  = df['text'].astype(str).apply(lambda x: len(x.split()))

print("TITLE WORD COUNT STATS:")
print(df.groupby('label')['title_word_count'].describe().round(1))
print("\nTEXT WORD COUNT STATS:")
print(df.groupby('label')['text_word_count'].describe().round(1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, color, name in [(0, '#E05C5C', 'Fake'), (1, '#5C9BE0', 'Real')]:
    subset = df[df['label'] == label]
    axes[0].hist(subset['title_word_count'].clip(0, 50), bins=30, alpha=0.6, color=color, label=name, edgecolor='white')
    axes[1].hist(subset['text_word_count'].clip(0, 1000), bins=50, alpha=0.6, color=color, label=name, edgecolor='white')
axes[0].set_title('Title Word Count Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Word Count (clipped at 50)')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[1].set_title('Article Body Word Count Distribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Word Count (clipped at 1000)')
axes[1].set_ylabel('Frequency')
axes[1].legend()
plt.tight_layout()
plt.savefig('text_length_distributions.png', dpi=150)
plt.show()

# ── Top Words per Class ──────────────────────────────────────
STOPWORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for',
    'of','with','is','it','this','that','was','are','be','as',
    'by','from','have','has','had','not','he','she','they',
    'we','you','i','his','her','their','its','our','said','will',
    'would','could','been','were','more','also','about','up',
    'which','who','all','one','than','when','out','so','what',
    'can','s','t','just','there','if','do','my','your'
])

def get_top_words(series, n=20):
    words = ' '.join(series.astype(str).str.lower()).split()
    words = [w for w in words if w.isalpha() and w not in STOPWORDS and len(w) > 2]
    return Counter(words).most_common(n)

fake_words = get_top_words(df[df['label'] == 0]['text'])
real_words = get_top_words(df[df['label'] == 1]['text'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, words, title, color in [
    (axes[0], fake_words, 'Top Words — Fake News', '#E05C5C'),
    (axes[1], real_words, 'Top Words — Real News', '#5C9BE0')
]:
    terms, counts = zip(*words)
    ax.barh(list(reversed(terms)), list(reversed(counts)), color=color, edgecolor='white')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency')
plt.tight_layout()
plt.savefig('top_words.png', dpi=150)
plt.show()

# ============================================================
# STEP 2 — Stratified Subset, Splits, Tokenization & GloVe
# ============================================================

# ── Create 20k Stratified Subset ─────────────────────────────
fake_sample = df[df['label'] == 0].sample(n=10000, random_state=42)
real_sample = df[df['label'] == 1].sample(n=10000, random_state=42)
df_subset = pd.concat([fake_sample, real_sample]).reset_index(drop=True)
df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Subset shape: {df_subset.shape}")
print(f"Label balance:\n{df_subset['label'].value_counts()}")

# ── Combine Title + Text ─────────────────────────────────────
df_subset['combined'] = (
    df_subset['title'].astype(str).str.strip()
    + " [SEP] "
    + df_subset['text'].astype(str).str.strip()
)
df_subset['combined_word_count'] = df_subset['combined'].apply(lambda x: len(x.split()))
print("Combined text word count stats:")
print(df_subset['combined_word_count'].describe().round(1))

# ── Train / Val / Test Split (70 / 15 / 15) ─────────────────
X = df_subset['combined'].values
y = df_subset['label'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train size:      {len(X_train):,}  ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation size: {len(X_val):,}   ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test size:       {len(X_test):,}   ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nTrain label balance:  {Counter(y_train)}")
print(f"Val label balance:    {Counter(y_val)}")
print(f"Test label balance:   {Counter(y_test)}")

# Save raw splits for DistilBERT
np.save('X_train.npy', X_train)
np.save('X_val.npy',   X_val)
np.save('X_test.npy',  X_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy',   y_val)
np.save('y_test.npy',  y_test)
df_subset.to_csv('welfake_subset_20k.csv', index=False)
print("Saved raw splits (.npy) and welfake_subset_20k.csv")

# ── Text Cleaning ────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'\[sep\]', ' sep ', text)          # preserve separator
    text = re.sub(r'[^a-z\s]', '', text)              # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()          # collapse whitespace
    return text

X_train_clean = np.array([clean_text(t) for t in X_train])
X_val_clean   = np.array([clean_text(t) for t in X_val])
X_test_clean  = np.array([clean_text(t) for t in X_test])

lengths = [len(t.split()) for t in X_train_clean]
p90     = np.percentile(lengths, 90)
MAX_LEN = 256
covered = sum(1 for l in lengths if l <= MAX_LEN) / len(lengths) * 100
print(f"Articles fully covered by MAX_LEN={MAX_LEN}: {covered:.1f}%")

plt.figure(figsize=(11, 4))
plt.hist(lengths, bins=80, color='#2E75B6', edgecolor='white', alpha=0.85)
plt.axvline(MAX_LEN, color='red',    linestyle='--', linewidth=2, label=f'MAX_LEN = {MAX_LEN} ({covered:.0f}% coverage)')
plt.axvline(p90,     color='orange', linestyle='--', linewidth=1.5, label=f'90th pct = {p90:.0f}')
plt.title('Cleaned Token Length Distribution — Training Set', fontsize=13, fontweight='bold')
plt.xlabel('Token Count (after cleaning)')
plt.ylabel('Number of Articles')
plt.legend()
plt.tight_layout()
plt.savefig('cleaned_lengths.png', dpi=150)
plt.show()

# ── Tokenization ─────────────────────────────────────────────
MAX_VOCAB = 30000
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_clean)
word_index = tokenizer.word_index
print(f"Full vocabulary size:   {len(word_index):,}")
print(f"Capped vocabulary size: {MAX_VOCAB:,}")

X_train_seq = tokenizer.texts_to_sequences(X_train_clean)
X_val_seq   = tokenizer.texts_to_sequences(X_val_clean)
X_test_seq  = tokenizer.texts_to_sequences(X_test_clean)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_val_pad   = pad_sequences(X_val_seq,   maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding='post', truncating='post')

print(f"X_train_pad: {X_train_pad.shape} | X_val_pad: {X_val_pad.shape} | X_test_pad: {X_test_pad.shape}")

# ── GloVe Embeddings ─────────────────────────────────────────
GLOVE_PATH = 'glove/glove.6B.100d.txt'
EMBED_DIM  = 100

if not os.path.exists(GLOVE_PATH):
    print("GloVe not found. Downloading...")
    os.makedirs('glove', exist_ok=True)
    os.system('wget -nc http://nlp.stanford.edu/data/glove.6B.zip -O glove.6B.zip')
    with zipfile.ZipFile('glove.6B.zip', 'r') as z:
        z.extractall('glove')
    print("GloVe downloaded and extracted.")

glove_index = {}
with open(GLOVE_PATH, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word   = values[0]
        vector = np.array(values[1:], dtype='float32')
        glove_index[word] = vector
print(f"GloVe loaded: {len(glove_index):,} vectors")

vocab_size       = min(MAX_VOCAB, len(word_index)) + 1
embedding_matrix = np.zeros((vocab_size, EMBED_DIM))
hits, misses = 0, 0

for word, idx in word_index.items():
    if idx >= vocab_size:
        continue
    vector = glove_index.get(word)
    if vector is not None:
        embedding_matrix[idx] = vector
        hits += 1
    else:
        misses += 1

print(f"GloVe coverage: {hits/(hits+misses)*100:.1f}%")

# ── Save All Preprocessed Artifacts ──────────────────────────
np.save('X_train_pad.npy',      X_train_pad)
np.save('X_val_pad.npy',        X_val_pad)
np.save('X_test_pad.npy',       X_test_pad)
np.save('embedding_matrix.npy', embedding_matrix)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\nAll preprocessing artifacts saved.")
print(f"CONFIG: MAX_LEN={MAX_LEN} | MAX_VOCAB={MAX_VOCAB} | EMBED_DIM={EMBED_DIM} | vocab_size={vocab_size}")
print("Preprocessing complete. Proceed to 02_model_bilstm.py")
