import argparse
import json
import pickle
import numpy as np
import torch

from deep_learning.rnn_model import RNNClassifier
from deep_learning.lstm_model import LSTMClassifier


def preprocess_input(text, vocab, max_len=50):
    tokens = [vocab.get(word, vocab["<unk>"]) for word in text.lower().split()]

    padded = np.zeros(max_len, dtype=int)

    if len(tokens) > 0:
        padded[:min(len(tokens), max_len)] = tokens[:max_len]

    return torch.LongTensor(padded).unsqueeze(0)


def load_model(model_name):

    # Load vocabulary
    with open(f"saved_models/{model_name}_vocab.json", "r") as f:
        vocab = json.load(f)

    # Load label encoder
    with open(f"saved_models/{model_name}_label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    vocab_size = len(vocab)
    num_classes = len(label_encoder.classes_)

    if model_name == "rnn":
        model = RNNClassifier(
            vocab_size,
            100,
            128,
            num_classes,
            2,
            True,
            0.5
        )

    elif model_name == "lstm":
        model = LSTMClassifier(
            vocab_size,
            100,
            128,
            num_classes,
            2,
            True,
            0.5
        )

    else:
        raise ValueError("Choose 'rnn' or 'lstm'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(
        torch.load(
            f"saved_models/{model_name}_sentiment_model.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, vocab, label_encoder, device


def predict(text, model_name):

    model, vocab, label_encoder, device = load_model(model_name)

    input_tensor = preprocess_input(text, vocab).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        _, predicted = torch.max(output, 1)

    sentiment = label_encoder.inverse_transform(
        predicted.cpu().numpy()
    )[0]

    return sentiment


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["rnn", "lstm"],
        required=True,
        help="Choose prediction model"
    )

    args = parser.parse_args()

    test_texts = [

        "satisfied application process unhappy documentation",

        "complained credit score error want manual review",

        "neutral application okay expects quicker disbursal",

        "happy with the service and quick response time",

        "unhappy with the delays and poor customer support"

    ]

    print("=" * 50)
    print(f"{args.model.upper()} Sentiment Prediction")
    print("=" * 50)

    for text in test_texts:

        sentiment = predict(text, args.model)

        print(f"Text      : {text}")
        print(f"Prediction: {sentiment}")
        print("-" * 50)

# python predict_model.py --model rnn
# python predict_model.py --model lstm