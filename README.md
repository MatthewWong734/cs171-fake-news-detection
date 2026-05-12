# CS 171 Final Project — Fake News Detection

**Team:** Madiha Fatima, Shivangi Dua, Matthew Wong  
**Course:** CS 171-01, Spring 2026

## What we built
We trained two models to classify news articles as real or fake — a BiLSTM 
we built from scratch and a fine-tuned DistilBERT. We used the WELFake dataset 
which has around 72k articles pulled from four different sources.

## How we split the work

| What | Who |
|---|---|
| Data preprocessing and cleaning pipeline | Shivangi |
| BiLSTM model architecture and training | Shivangi |
| DistilBERT fine-tuning with HuggingFace | Madiha |
| Model testing and evaluation runs | Matthew |
| Results analysis, plots, confusion matrices | Shivangi |
| Report writing | All members |

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| BiLSTM | 96.27% | 96.26% | 99.21% |
| DistilBERT | 98.13% | 98.12% | 99.74% |

DistilBERT did better pretty much across the board, which makes sense since 
it came in already knowing language from pretraining. The BiLSTM had to learn 
everything from the training data with only GloVe embeddings as a starting point.

## Stack
Python, Pandas, NumPy, Sklearn, TensorFlow/Keras, PyTorch, HuggingFace Transformers
