# 🏦 Intelligent Loan Intelligence Platform using NLP

An end-to-end AI-powered Loan Intelligence Platform built using **Natural Language Processing, Machine Learning, Deep Learning, Transformers, FastAPI, and Streamlit**.

The platform analyzes customer loan applications, feedback, and agent remarks to perform:

- Loan Status Prediction
- Customer Sentiment Analysis
- Named Entity Recognition (NER)
- Semantic Search
- Question Answering
- Similar Customer Retrieval
- Loan Analytics Dashboard

---

# 📌 Project Overview

Traditional loan processing mainly relies on structured information such as CIBIL score, employment status, and income. However, valuable information also exists inside textual data like customer applications, feedback, and agent remarks.

This project extracts meaningful insights from these text fields using modern NLP techniques and provides intelligent APIs for banking applications.

---

# 🚀 Features

## NLP Preprocessing

- Lowercasing
- Regex Cleaning
- Stopword Removal
- Tokenization
- Punctuation Removal
- Number Removal
- Lemmatization
- Stemming

---

## Exploratory NLP Analysis

- Word Frequency
- Vocabulary Statistics
- Word Clouds
- Bigrams
- Trigrams

---

## Feature Engineering

- Count Vectorizer
- TF-IDF
- One Hot Encoding
- Feature Dimension Comparison

---

## Word Embeddings

- Word2Vec (CBOW)
- Word2Vec (Skip-Gram)
- FastText
- GloVe

Comparison includes:

- Vocabulary Coverage
- OOV Handling
- Similarity Performance

---

## Classical Machine Learning Models

Customer Sentiment Classification

- Logistic Regression
- Naive Bayes
- Random Forest
- Linear SVM

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Named Entity Recognition

Extracts

- Person
- Organization
- Location
- Dates
- Money

using **spaCy**

---

## Semantic Search

Uses

- Sentence Transformers
- Cosine Similarity

Supports queries such as

- Education loan requests
- Dissatisfied customers
- Home renovation applications
- Delayed approvals
- Positive customer experiences

Returns Top-K most similar applications.

---

## Resume-style Semantic Retrieval

Each customer application is indexed as a document.

Supports

- Semantic Retrieval
- Vector Indexing
- Similar Customer Search

---

## Deep Learning Models

### RNN

Predicts Customer Sentiment.

### LSTM

Compares performance with RNN.

Evaluation

- Accuracy
- Loss
- Precision
- Recall

---

## Transformer Model

Fine-tuned BERT for Loan Status Prediction.

Evaluation

- Accuracy
- F1 Score
- Confusion Matrix

---

## Question Answering System

Built using Hugging Face Transformers.

Example Questions

- Why was my loan rejected?
- Which applicants have low credit history?
- Show customers requesting business loans.
- What documents are required?

---

# 📁 Project Structure

```
Loan-NLP-System/
│
├── api/
│   ├── main.py
│
├── data/
│
├── notebooks/
│
├── preprocessing/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│
├── embeddings/
│   ├── embedding_models.py
│
├── models/
│   ├── classical_ml/
│   │      ├── sentiment.py
│   │      ├── ner.py
│   │      ├── semantic_search.py
│   │
│   ├── deep_learning/
│   │      ├── rnn.py
│   │      ├── lstm.py
│   │
│   ├── transformers/
│          ├── bert_classifier.py
│          ├── qa_engine.py
│
├── saved_models/
│
├── streamlit/
│   └── app.py
│
├── utils/
│
├── requirements.txt
│
└── README.md
```


