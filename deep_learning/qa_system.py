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



# Structured Question Handler


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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Precompute TF-IDF matrix for context retrieval
app_texts = df["Application_Text"].fillna("").tolist()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(app_texts)

def get_relevant_context(question, top_n=3):
    q_vec = tfidf.transform([question])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[-top_n:][::-1]
    context_parts = []
    for idx in top_indices:
        if sims[idx] > 0:
            context_parts.append(app_texts[idx])
    return "\n".join(context_parts)

# -------------------------------------------------
# BERT Question Answering
# -------------------------------------------------

def bert_answer(question):

    context = get_relevant_context(question, top_n=1)
    # Truncate context to ~1500 characters to fit within 512 token limit
    if len(context) > 1500:
        context = context[:1500] + "..."
    
    prompt = f"""
Answer the following question based ONLY on the given context. If the answer is not in the context, answer based on your knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    result = qa_pipeline(
        prompt,
        max_new_tokens=150
    )

    return result[0]["generated_text"]


# -------------------------------------------------
# Main Query Function
# -------------------------------------------------

def ask_question(question):

    result = structured_query(question)

    if result is not None:
        if result.empty:
            return "No matching records found."
        return result.head(10).to_string(index=False)

    return bert_answer(question)


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