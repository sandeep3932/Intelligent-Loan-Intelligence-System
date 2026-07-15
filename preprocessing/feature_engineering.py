import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

def main():
    print("=========================================")
    print("Starting Engineer 1: Task 3 (Feature Engineering)")
    print("=========================================")

    # Load the real processed dataset
    dataset_path = 'data/processed_dataset.csv'
    if not os.path.exists(dataset_path):
        dataset_path = 'hdfc_loan_dataset_full_enriched_fixed_v2.csv'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Processed dataset not found at {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Input dataset shape: {df.shape}")

    # Processed Text Columns
    text_cols = ['Application_Text_processed', 'Customer_Feedback_processed', 'Agent_Notes_processed']
    # Categorical Columns (as specified in PDF)
    cat_cols = ['Education', 'Employment_Status', 'Purpose_of_Loan', 'Occupation', 'State', 'City', 'Credit_History', 'Property_Area']

    # Combine text columns into a single corpus for vectorization
    corpus = df[text_cols].fillna('').agg(' '.join, axis=1).tolist()
    
    os.makedirs('data/features', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    # ========================================
    # Task 3.1: CountVectorizer
    # ========================================
    print("\n--- Task 3.1: CountVectorizer ---")
    count_vec = CountVectorizer(max_features=5000)
    cv_features = count_vec.fit_transform(corpus)
    joblib.dump(count_vec, 'saved_models/count_vectorizer.pkl')
    sp.save_npz('data/features/count_vectorizer_features.npz', cv_features)
    print(f"CountVectorizer features generated and saved to data/features/count_vectorizer_features.npz")

    # ========================================
    # Task 3.2: TF-IDF
    # ========================================
    print("\n--- Task 3.2: TF-IDF ---")
    tfidf_vec = TfidfVectorizer(max_features=5000)
    tfidf_features = tfidf_vec.fit_transform(corpus)
    joblib.dump(tfidf_vec, 'saved_models/tfidf_vectorizer.pkl')
    sp.save_npz('data/features/tfidf_features.npz', tfidf_features)
    print(f"TF-IDF features generated and saved to data/features/tfidf_features.npz")

    # ========================================
    # Task 3.3: One-Hot Encoding
    # ========================================
    print("\n--- Task 3.3: One-Hot Encoding ---")
    df_cat = df[cat_cols].fillna("Unknown")
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    ohe_features = ohe.fit_transform(df_cat)
    joblib.dump(ohe, 'saved_models/one_hot_encoder.pkl')
    sp.save_npz('data/features/one_hot_features.npz', ohe_features)
    print(f"One-Hot Encoding features generated and saved to data/features/one_hot_features.npz")

    # ========================================
    # Task 3.4: Compare Feature Dimensions
    # ========================================
    print("\n--- Task 3.4: Compare Feature Dimensions ---")
    print(f"Original Dataset Rows           : {df.shape[0]}")
    print(f"CountVectorizer Matrix Shape    : {cv_features.shape}")
    print(f"TF-IDF Matrix Shape             : {tfidf_features.shape}")
    print(f"One-Hot Encoding Matrix Shape   : {ohe_features.shape}")
    
    # Ensure exact row alignment
    assert df.shape[0] == cv_features.shape[0] == tfidf_features.shape[0] == ohe_features.shape[0], "Row alignment mismatch!"
    print("Row alignment verified: All feature matrices match the original dataset row count.")

    print("\n=========================================")
    print("Task 3 Completed Successfully.")
    print("=========================================")

if __name__ == "__main__":
    main()
