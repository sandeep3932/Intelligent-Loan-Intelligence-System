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

## How to Run the Application

The application is split into a **FastAPI backend** and a **Streamlit frontend**. You need to run both simultaneously in separate terminal windows.

### 1. Start the FastAPI Backend
Open your terminal, ensure your virtual environment is activated, and navigate to the `api` directory:
```bash
cd api
python -m uvicorn main:app --reload
```
*The API will start at `http://localhost:8000`. You can view the Swagger UI documentation at `http://localhost:8000/docs`.*

### 2. Start the Streamlit Dashboard
Open a **new** terminal window, activate your virtual environment, and navigate to the `streamlit` directory:
```bash
cd streamlit
python -m streamlit run app.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`. If you are running on a server, you might need to use `--server.headless true`.*
gitignore
