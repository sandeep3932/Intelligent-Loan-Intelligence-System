import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deep_learning.rnn_model import RNNClassifier
from deep_learning.lstm_model import LSTMClassifier


# ==========================
# Reproducibility
# ==========================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==========================
# Tokenization
# ==========================
def tokenize(text, vocab):
    return [vocab.get(word, vocab["<unk>"]) for word in text.split()]


def pad_sequence(sequences, max_len):
    padded = np.zeros((len(sequences), max_len), dtype=int)

    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue

        if len(seq) <= max_len:
            padded[i, :len(seq)] = seq
        else:
            padded[i] = seq[:max_len]

    return padded


# ==========================
# Main
# ==========================
def main(args):

    print("=" * 50)
    print(f"Training {args.model.upper()} Sentiment Model")
    print("=" * 50)

    # --------------------------
    # Load Dataset
    # --------------------------
    dataset_path = "data/processed_dataset.csv"

    if not os.path.exists(dataset_path):
        print(f"{dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path)

    df = df[
        [
            "Customer_Feedback_processed",
            "Customer_Sentiment"
        ]
    ].dropna()

    print(f"Dataset Size : {len(df)}")

    # --------------------------
    # Encode Labels
    # --------------------------
    label_encoder = LabelEncoder()

    df["label"] = label_encoder.fit_transform(
        df["Customer_Sentiment"]
    )

    num_classes = len(label_encoder.classes_)

    print("Classes :", label_encoder.classes_)

    # --------------------------
    # Vocabulary
    # --------------------------
    all_text = " ".join(
        df["Customer_Feedback_processed"].astype(str)
    ).split()

    unique_words = sorted(set(all_text))

    vocab = {
        word: idx + 2
        for idx, word in enumerate(unique_words)
    }

    vocab["<pad>"] = 0
    vocab["<unk>"] = 1

    vocab_size = len(vocab)

    print("Vocabulary Size :", vocab_size)

    # --------------------------
    # Tokenize
    # --------------------------
    MAX_LEN = 50

    df["tokens"] = df[
        "Customer_Feedback_processed"
    ].apply(
        lambda x: tokenize(str(x), vocab)
    )

    X = pad_sequence(
        df["tokens"].values,
        MAX_LEN
    )

    y = df["label"].values

    # --------------------------
    # Train Test Split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    # --------------------------
    # DataLoader
    # --------------------------
    train_loader = DataLoader(
        TensorDataset(
            torch.LongTensor(X_train),
            torch.LongTensor(y_train)
        ),
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.LongTensor(X_test),
            torch.LongTensor(y_test)
        ),
        batch_size=32
    )

    # --------------------------
    # Model
    # --------------------------
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5

    if args.model == "rnn":

        model = RNNClassifier(
            vocab_size,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            num_classes,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT
        )

    else:

        model = LSTMClassifier(
            vocab_size,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            num_classes,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using Device :", device)

    model.to(device)

    # --------------------------
    # Loss & Optimizer
    # --------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # --------------------------
    # Training
    # --------------------------
    EPOCHS = 10

    best_val_acc = 0

    print("\nTraining Started...\n")

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            train_total += batch_y.size(0)

            train_correct += (
                predicted == batch_y
            ).sum().item()

        train_acc = 100 * train_correct / train_total

        # --------------------------
        # Validation
        # --------------------------
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for batch_x, batch_y in test_loader:

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)

                _, predicted = torch.max(outputs, 1)

                val_total += batch_y.size(0)

                val_correct += (
                    predicted == batch_y
                ).sum().item()

        val_acc = 100 * val_correct / val_total

        print(
            f"Epoch {epoch+1}/{EPOCHS}"
            f" | Loss: {train_loss/len(train_loader):.4f}"
            f" | Train Acc: {train_acc:.2f}%"
            f" | Val Acc: {val_acc:.2f}%"
        )

        # --------------------------
        # Save Best Model
        # --------------------------
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            os.makedirs(
                "saved_models",
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                f"saved_models/{args.model}_sentiment_model.pth"
            )

    # --------------------------
    # Save Artifacts
    # --------------------------
    with open(
        f"saved_models/{args.model}_label_encoder.pkl",
        "wb"
    ) as f:

        pickle.dump(label_encoder, f)

    with open(
        f"saved_models/{args.model}_vocab.json",
        "w"
    ) as f:

        json.dump(vocab, f)

    print("\nTraining Complete")
    print(f"Best Validation Accuracy : {best_val_acc:.2f}%")
    print("Artifacts Saved Successfully.")


# ==========================
# Entry
# ==========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["rnn", "lstm"],
        required=True,
        help="Choose model to train"
    )

    args = parser.parse_args()

    main(args)


# python train_model.py --model rnn

# python train_model.py --model lstm