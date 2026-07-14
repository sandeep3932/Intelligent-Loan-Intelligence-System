import torch
import json
import pickle
import numpy as np
from deep_learning.rnn_model import RNNClassifier

def preprocess_input(text, vocab, max_len=50):
    tokens = [vocab.get(word, vocab['<unk>']) for word in text.lower().split()]
    padded = np.zeros(max_len, dtype=int)
    if len(tokens) > 0:
        if len(tokens) <= max_len:
            padded[:len(tokens)] = tokens
        else:
            padded[:] = tokens[:max_len]
    return torch.LongTensor(padded).unsqueeze(0) # Add batch dimension

def predict(text):
    # 1. Load Artifacts
    with open('saved_models/rnn_vocab.json', 'r') as f:
        vocab = json.load(f)
    
    with open('saved_models/sentiment_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    vocab_size = len(vocab)
    num_classes = len(label_encoder.classes_)
    
    # 2. Initialize Model
    model = RNNClassifier(vocab_size, 100, 128, num_classes, 2, True, 0.5)
    model.load_state_dict(torch.load('saved_models/rnn_sentiment_model.pth'))
    model.eval()
    
    # 3. Predict
    input_tensor = preprocess_input(text, vocab)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        sentiment = label_encoder.inverse_transform(predicted.numpy())[0]
        
    return sentiment

if __name__ == "__main__":
    test_texts = [
        "satisfied application process unhappy documentation",
        "complained credit score error want manual review",
        "neutral application okay expects quicker disbursal",
        "happy with the service and quick response time",
        "unhappy with the delays and poor customer support"
    ]
    
    print("=========================================")
    print("RNN Sentiment Prediction Test")
    print("=========================================")
    for text in test_texts:
        sentiment = predict(text)
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {sentiment}\n")
    print("=========================================")
