# Fake News Detection — BiLSTM vs DistilBERT
**CS171 Project | WELFake Dataset**

## Project Overview
Binary classification of fake vs. real news articles using two architectures:
- **BiLSTM** with GloVe embeddings (traditional deep learning)
- **DistilBERT** fine-tuned for sequence classification (transformer-based)

## Results Summary

| Metric     | BiLSTM  | DistilBERT | Delta  |
|------------|---------|------------|--------|
| Accuracy   | 96.27%  | 99.07%     | +2.80% |
| Precision  | 94.95%  | 98.36%     | +3.41% |
| Recall     | 97.73%  | 99.80%     | +2.07% |
| F1-Score   | 96.32%  | 99.07%     | +2.75% |
| ROC-AUC    | 99.47%  | 99.97%     | +0.50% |

## File Structure

```
├── 01_data_preprocessing.py   # EDA, cleaning, tokenization, GloVe embeddings, train/val/test split
├── 02_model_bilstm.py         # BiLSTM model architecture + training
├── 03_model_distilbert.py     # DistilBERT fine-tuning
├── 04_evaluation.py           # Test set evaluation for both models
├── 05_results_analysis.py     # Side-by-side comparison plots and error analysis
└── README.md
```

## How to Run (Google Colab)

**Prerequisites:** Upload `WELFake_Dataset.csv` to your Colab environment.

Run files in order:
```
01_data_preprocessing.py   →  generates X_train/val/test splits, embedding_matrix, tokenizer
02_model_bilstm.py         →  trains BiLSTM, saves bilstm_best.keras
03_model_distilbert.py     →  fine-tunes DistilBERT, saves distilbert_best/  (requires GPU)
04_evaluation.py           →  evaluates both models on test set, saves metrics
05_results_analysis.py     →  generates all comparison plots
```

## Dataset
**WELFake** — 72,134 news articles (Fake + Real), merged from four public datasets.
- Labels: `0 = Fake`, `1 = Real`
- We use a balanced 20,000-article stratified subset (10k fake + 10k real)
- Split: 70% train / 15% val / 15% test

## Model Details

### BiLSTM
- GloVe 6B 100d embeddings (frozen)
- Bidirectional LSTM (128 units per direction)
- Dropout: 0.4 | Spatial Dropout: 0.2
- Optimizer: Adam (lr=1e-3) | Batch size: 64

### DistilBERT
- `distilbert-base-uncased` fine-tuned for binary classification
- Max sequence length: 256 tokens
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Linear warmup scheduler (10% warmup) | 4 epochs | Batch size: 16

## Dependencies
```
tensorflow>=2.0
torch
transformers
scikit-learn
pandas
numpy
matplotlib
seaborn
```
