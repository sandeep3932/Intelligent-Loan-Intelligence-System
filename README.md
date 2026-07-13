# Intelligent Loan Intelligence Platform using NLP, Machine Learning, Deep Learning & Transformers

An AI-powered Loan Intelligence Platform using NLP, Machine Learning, Deep Learning, and Transformer models for sentiment analysis, semantic search, loan classification, NER, and question answering.

## Project Overview
This project builds an end-to-end NLP system capable of understanding customer applications, feedback, agent remarks, and semantic queries from a banking loan dataset. 

It is divided among three engineers:
- **Engineer 1:** NLP Preprocessing & Feature Engineering
- **Engineer 2:** NLP Intelligence Models
- **Engineer 3:** Deep Learning & Transformer Models

Currently, the foundation for **Engineer 1** has been implemented, including robust text preprocessing pipelines, exploratory data analysis, traditional feature engineering (CountVectorizer, TF-IDF, One-Hot Encoding), and advanced word embeddings (Word2Vec, FastText, GloVe).

## Folder Structure
```
Loan-NLP-System/
│
├── data/                    # Contains processed datasets and engineered feature matrices (.npz)
├── notebooks/               # Jupyter notebooks for interactive analysis
├── preprocessing/           # Preprocessing and Feature Engineering scripts
│   ├── preprocessing.py
│   └── feature_engineering.py
├── embeddings/              # Embedding model scripts
│   └── embedding_models.py
├── models/                  # Trained Machine Learning and Deep Learning models
│   ├── classical_ml/
│   ├── deep_learning/
│   └── transformers/
├── api/                     # FastAPI deployment (To be implemented)
├── streamlit/               # Streamlit dashboard (To be implemented)
├── utils/                   # Shared utility scripts
├── saved_models/            # Trained embedding models and vectorizers
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```
