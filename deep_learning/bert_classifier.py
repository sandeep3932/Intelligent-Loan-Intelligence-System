import os
import pickle
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 2
EPOCHS = 8
LEARNING_RATE = 2e-5


class LoanDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def evaluate(model, loader):

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for batch in loader:

            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=ids,
                attention_mask=mask
            )

            pred = torch.argmax(outputs.logits, dim=1)

            predictions.extend(pred.cpu().numpy())
            actual.extend(labels.cpu().numpy())

    accuracy = accuracy_score(actual, predictions)

    f1 = f1_score(
        actual,
        predictions,
        average="weighted"
    )

    return accuracy, f1, actual, predictions


def main():

    print("="*50)
    print("BERT Loan Status Classification")
    print("="*50)

    df = pd.read_csv("data/processed_dataset.csv")

    df = df[
        ["Application_Text_BERT","Loan_Status"]
    ].dropna()

    encoder = LabelEncoder()

    df["label"] = encoder.fit_transform(df["Loan_Status"])

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["Application_Text_BERT"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = LoanDataset(
        train_texts.tolist(),
        train_labels.tolist(),
        tokenizer
    )

    test_dataset = LoanDataset(
        test_texts.tolist(),
        test_labels.tolist(),
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(encoder.classes_)
    ).to(DEVICE)

    # ----------------------------
    # Loss Function (Weighted)
    # ----------------------------

    class_counts = np.bincount(train_labels)

    class_weights = torch.tensor(
        [
            len(train_labels) / (2 * class_counts[0]),
            len(train_labels) / (2 * class_counts[1])
        ],
        dtype=torch.float
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_f1 = 0.0

    print("\nTraining Started...\n")

    # ----------------------------
    # Training Loop
    # ----------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(
                outputs.logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            scheduler.step()

            total_loss += loss.item()

        accuracy, f1, _, _ = evaluate(
            model,
            test_loader
        )

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"F1: {f1:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1

            os.makedirs(
                "saved_models/bert_classifier",
                exist_ok=True
            )

            model.save_pretrained(
                "saved_models/bert_classifier"
            )

            tokenizer.save_pretrained(
                "saved_models/bert_classifier"
            )

            with open(
                "saved_models/bert_classifier/label_encoder.pkl",
                "wb"
            ) as f:
                pickle.dump(
                    encoder,
                    f
                )

    print("\nTraining Finished")
    print(f"Best F1 Score : {best_f1:.4f}")

    # ----------------------------
    # Final Evaluation
    # ----------------------------

    accuracy, f1, y_true, y_pred = evaluate(
        model,
        test_loader
    )

    print("\nFinal Results")
    print("=" * 40)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nConfusion Matrix\n")

    print(
        pd.DataFrame(
            cm,
            index=encoder.classes_,
            columns=encoder.classes_
        )
    )

    print("\nModel Saved Successfully")


if __name__ == "__main__":
    main()
