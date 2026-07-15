import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("=" * 60)
print("Loading Question Answering Model...")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

def qa_pipeline(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return [{"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}]

print("Model Loaded Successfully!\n")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_csv("data/processed_dataset.csv")


# -------------------------------------------------
# Structured Question Handler
# -------------------------------------------------

def structured_query(question):

    question = question.lower()

    # --------------------------------------------
    # Low Credit History
    # --------------------------------------------

    if "low credit history" in question:

        result = df[df["Credit_History"] == 0][
            [
                "Customer_Name",
                "Loan_Status",
                "CIBIL_Score"
            ]
        ]

        return result

    # --------------------------------------------
    # Business Loan
    # --------------------------------------------

    elif "business loan" in question:

        result = df[
            df["Purpose_of_Loan"]
            .str.contains("Business", case=False)
        ][
            [
                "Customer_Name",
                "Purpose_of_Loan",
                "Loan_Amount"
            ]
        ]

        return result

    # --------------------------------------------
    # Collateral
    # --------------------------------------------

    elif "collateral" in question:

        result = df[
            df["Application_Text"]
            .str.contains(
                "collateral",
                case=False,
                na=False
            )
        ][
            [
                "Customer_Name",
                "Application_Text"
            ]
        ]

        return result

    return None


# -------------------------------------------------
# BERT Question Answering
# -------------------------------------------------

def bert_answer(question):

    context = " ".join(
        df["Application_Text"].astype(str).tolist()
    )[:3000]

    prompt = f"""
Answer the following question using the context.

Context:
{context}

Question:
{question}

Answer:
"""

    result = qa_pipeline(
        prompt,
        max_new_tokens=50
    )

    return result[0]["generated_text"]


# -------------------------------------------------
# Main Query Function
# -------------------------------------------------

def ask_question(question):

    result = structured_query(question)

    if result is not None:

        print("\nStructured Query Result\n")

        print(result.head(20))

        return

    print("\nBERT Answer\n")

    print(bert_answer(question))


# -------------------------------------------------
# Demo
# -------------------------------------------------

if __name__ == "__main__":

    questions = [

        "Why was my loan rejected?",

        "What documents are required?",

        "Which applicants have low credit history?",

        "Which customer requested a business loan?",

        "Show applications mentioning collateral."

    ]

    for q in questions:

        print("=" * 60)

        print("Question:")

        print(q)

        ask_question(q)

        print()