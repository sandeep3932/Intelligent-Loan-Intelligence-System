from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import sys
import os

# Add parent directory to path so we can import from deep_learning etc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict_model import load_model as load_rnn_model
from predict_model import preprocess_input as preprocess_rnn_input
from deep_learning.qa_system import qa_pipeline

from deep_learning.lstm_model import LSTMClassifier
import json

app = FastAPI(title="Loan Intelligence API")

# Load models at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. RNN Sentiment (LSTM Actually)
print("Loading RNN Sentiment Model...")
with open('saved_models/rnn_vocab.json', 'r') as f:
    rnn_vocab = json.load(f)
with open('saved_models/sentiment_label_encoder.pkl', 'rb') as f:
    rnn_label_encoder = pickle.load(f)

vocab_size = len(rnn_vocab)
num_classes = len(rnn_label_encoder.classes_)
rnn_model = LSTMClassifier(vocab_size, 100, 128, num_classes, 2, True, 0.5)
rnn_model.load_state_dict(torch.load('saved_models/rnn_sentiment_model.pth', map_location=device))
rnn_model.to(device)
rnn_model.eval()
rnn_device = device

# 2. BERT Classifier
print("Loading BERT Classifier Model...")
bert_tokenizer = BertTokenizer.from_pretrained("saved_models/bert_classifier")
bert_model = BertForSequenceClassification.from_pretrained("saved_models/bert_classifier").to(device)
with open("saved_models/bert_classifier/label_encoder.pkl", "rb") as f:
    bert_label_encoder = pickle.load(f)

# QA is already loaded when deep_learning.qa_system is imported

class TextRequest(BaseModel):
    text: str

class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/predict-sentiment")
def predict_sentiment(req: TextRequest):
    input_tensor = preprocess_rnn_input(req.text, rnn_vocab).to(rnn_device)
    with torch.no_grad():
        output = rnn_model(input_tensor)
        _, predicted = torch.max(output, 1)
    sentiment = rnn_label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    return {"sentiment": sentiment}

@app.post("/classify-loan")
def classify_loan(req: TextRequest):
    inputs = bert_tokenizer(req.text, return_tensors="pt", truncation=True, max_length=256, padding="max_length").to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, predicted = torch.max(probs, 1)
    status = bert_label_encoder.inverse_transform(predicted.cpu().numpy())[0]
    return {"loan_status": status, "confidence": float(conf.cpu().numpy())}

@app.post("/question-answer")
def question_answer(req: TextRequest):
    from deep_learning.qa_system import ask_question
    result = ask_question(req.text)
    return {"answer": result}

@app.post("/semantic-search")
def semantic_search(req: SemanticSearchRequest):
    # Placeholder for Engineer 2
    return {"results": [{"text": f"Dummy application matching '{req.query}'", "score": 0.95}]}

@app.post("/named-entities")
def named_entities(req: TextRequest):
    # Placeholder for Engineer 2
    return {"entities": [{"text": "John Doe", "label": "PERSON"}]}

@app.post("/similar-customers")
def similar_customers(req: TextRequest):
    # Placeholder for Engineer 2
    return {"similar_customers": ["Customer A", "Customer B"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
