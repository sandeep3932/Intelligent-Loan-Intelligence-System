import os
import json
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from deep_learning.rnn_model import RNNClassifier
from deep_learning.lstm_model import LSTMClassifier


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def tokenize(text, vocab):
    return [vocab.get(word, vocab["<unk>"]) for word in text.split()]


def pad_sequence(sequences, max_len):
    padded = np.zeros((len(sequences), max_len), dtype=int)

    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            padded[i, :min(len(seq), max_len)] = seq[:max_len]

    return padded


def evaluate_model(model_name, model_class):

    print(f"\nEvaluating {model_name.upper()}...")

    # --------------------------
    # Load dataset
    # --------------------------

    df = pd.read_csv("data/processed_dataset.csv")
    
    df = df[
        [
            "Customer_Feedback_processed",
            "Customer_Sentiment"
        ]
    ].dropna()

    # --------------------------
    # Load artifacts
    # --------------------------


    with open(f"saved_models/{model_name}_vocab.json") as f:
        vocab = json.load(f)

    with open(f"saved_models/{model_name}_label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    df["tokens"] = df["Customer_Feedback_processed"].apply(
        lambda x: tokenize(str(x), vocab)
    )

    X = pad_sequence(df["tokens"], 50)

    y = label_encoder.transform(df["Customer_Sentiment"])

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class(
        len(vocab),
        100,
        128,
        len(label_encoder.classes_),
        2,
        True,
        0.5
    )

    model.load_state_dict(
        torch.load(
            f"saved_models/{model_name}_sentiment_model.pth",
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    criterion = nn.CrossEntropyLoss()

    X_test = torch.LongTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    with torch.no_grad():

        outputs = model(X_test)

        loss = criterion(outputs, y_test_tensor)

        _, predictions = torch.max(outputs, 1)

    y_pred = predictions.cpu().numpy()

    metrics = {

        "Accuracy": accuracy_score(y_test, y_pred),

        "Loss": loss.item(),

        "Precision": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    return metrics, report, cm


def main():

    rnn_metrics, rnn_report, rnn_cm = evaluate_model(
        "rnn",
        RNNClassifier
    )

    lstm_metrics, lstm_report, lstm_cm = evaluate_model(
        "lstm",
        LSTMClassifier
    )

    comparison = pd.DataFrame({

        "Metric": list(rnn_metrics.keys()),

        "RNN": list(rnn_metrics.values()),

        "LSTM": list(lstm_metrics.values())

    })

    print("\n==============================")
    print("RNN vs LSTM Comparison")
    print("==============================")

    print(comparison)

    os.makedirs("results", exist_ok=True)

    comparison.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    print("\nClassification Report (RNN)")
    print(rnn_report)

    print("\nConfusion Matrix (RNN)")
    print(rnn_cm)

    print("\nClassification Report (LSTM)")
    print(lstm_report)

    print("\nConfusion Matrix (LSTM)")
    print(lstm_cm)

    print("\nComparison saved to results/model_comparison.csv")



if __name__ == "__main__":
    main()