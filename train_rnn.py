import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import json
from deep_learning.rnn_model import RNNClassifier

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def tokenize(text, vocab):
    """Converts text to a list of word indices based on vocab."""
    return [vocab.get(word, vocab['<unk>']) for word in text.split()]

def pad_sequence(sequences, max_len):
    """Pads or truncates sequences to max_len."""
    padded = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            if len(seq) <= max_len:
                padded[i, :len(seq)] = seq
            else:
                padded[i, :] = seq[:max_len]
    return padded

def main():
    print("=========================================")
    print("Starting RNN Sentiment Model Training")
    print("=========================================")

    # 1. Load Data
    dataset_path = 'data/processed_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run preprocessing first.")
        return

    df = pd.read_csv(dataset_path)
    # Use pre-processed text and sentiment target
    df = df[['Customer_Feedback_processed', 'Customer_Sentiment']].dropna()
    
    print(f"Dataset loaded. Rows: {len(df)}")

    # 2. Encode Target Labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Customer_Sentiment'])
    num_classes = len(label_encoder.classes_)
    print(f"Sentiment classes: {label_encoder.classes_}")

    # 3. Build Vocabulary
    all_text = ' '.join(df['Customer_Feedback_processed'].astype(str)).split()
    unique_words = sorted(list(set(all_text)))
    vocab = {word: i + 2 for i, word in enumerate(unique_words)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # 4. Preprocess Sequences
    MAX_LEN = 50
    df['tokenized'] = df['Customer_Feedback_processed'].apply(lambda x: tokenize(str(x), vocab))
    X = pad_sequence(df['tokenized'].values, MAX_LEN)
    y = df['label'].values

    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # 6. Create DataLoaders
    BATCH_SIZE = 32
    train_data = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    test_data = TensorDataset(torch.LongTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # 7. Initialize Model
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    model = RNNClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 8. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 9. Training Loop
    EPOCHS = 10
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                _, predicted = torch.max(predictions.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 10. Save Model and Artifacts
    os.makedirs('saved_models', exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), 'saved_models/rnn_sentiment_model.pth')
    
    # Save label encoder
    with open('saved_models/sentiment_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
        
    # Save vocabulary
    with open('saved_models/rnn_vocab.json', 'w') as f:
        json.dump(vocab, f)
        
    print("\nModel and artifacts saved to saved_models/")
    print("=========================================")

if __name__ == "__main__":
    main()
