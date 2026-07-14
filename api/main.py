from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import spacy
import sys
import os

# Add root directory to python path to import models
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from models.semantic_search import SemanticSearchEngine

app = FastAPI(title="Intelligent Loan Intelligence Platform APIs", description="APIs for NLP banking tasks.")

# Global variables for models
sentiment_model = None
sentiment_vectorizer = None
sentiment_le = None
nlp_ner = None
semantic_search_engine = None

@app.on_event("startup")
def load_models():
    global sentiment_model, sentiment_vectorizer, sentiment_le, nlp_ner, semantic_search_engine
    
    print("Loading Classical ML Sentiment Models...")
    try:
        model_dir = os.path.join(root_dir, 'saved_models', 'classical_ml')
        sentiment_model = joblib.load(os.path.join(model_dir, 'sentiment_model.pkl'))
        sentiment_vectorizer = joblib.load(os.path.join(model_dir, 'sentiment_vectorizer.pkl'))
        sentiment_le = joblib.load(os.path.join(model_dir, 'sentiment_label_encoder.pkl'))
    except Exception as e:
        print(f"Failed to load sentiment models: {e}")

    print("Loading spaCy NER Model...")
    try:
        nlp_ner = spacy.load("en_core_web_md")
    except Exception as e:
        print(f"Failed to load spaCy model: {e}")

    print("Loading Semantic Search Engine...")
    try:
        semantic_search_engine = SemanticSearchEngine()
        index_path = os.path.join(root_dir, 'saved_models', 'semantic_index_v2.pkl')
        if not semantic_search_engine.load_index(index_path):
            print("Warning: Semantic index not found. Semantic search might fail until index is built.")
    except Exception as e:
        print(f"Failed to load Semantic Search Engine: {e}")

    print("All models loaded successfully.")


# Schemas
class TextRequest(BaseModel):
    text: str

class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Loan NLP APIs are running."}

@app.post("/predict-sentiment")
def predict_sentiment(req: TextRequest):
    if sentiment_model is None:
        raise HTTPException(status_code=500, detail="Sentiment model not loaded.")
    
    # Preprocess text (assumes input text needs identical processing, we'll just vectorize directly)
    vec_text = sentiment_vectorizer.transform([req.text])
    pred_idx = sentiment_model.predict(vec_text)[0]
    sentiment = sentiment_le.inverse_transform([pred_idx])[0]
    
    # Attempt to get probability if supported
    confidence = None
    if hasattr(sentiment_model, "predict_proba"):
        probs = sentiment_model.predict_proba(vec_text)[0]
        confidence = float(max(probs))
        
    return {
        "text": req.text,
        "sentiment": sentiment,
        "confidence": confidence
    }

@app.post("/named-entities")
def extract_entities(req: TextRequest):
    if nlp_ner is None:
        raise HTTPException(status_code=500, detail="NER model not loaded.")
    
    doc = nlp_ner(req.text)
    allowed_labels = {'PERSON', 'ORG', 'LOC', 'GPE', 'DATE', 'MONEY'}
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ in allowed_labels]
    
    return {
        "text": req.text,
        "entities": entities
    }

@app.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest):
    if semantic_search_engine is None or semantic_search_engine.document_embeddings is None:
        raise HTTPException(status_code=500, detail="Semantic search index not loaded.")
    
    results = semantic_search_engine.search(req.query, top_k=req.top_k)
    # Convert numpy types to native python types for JSON serialization
    for r in results:
        r['score'] = float(r['score'])
        
    return {
        "query": req.query,
        "results": results
    }
