import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

def load_data(filepath):
    df = pd.read_csv(filepath)
    # Combine relevant text columns into a single document for better semantic matching
    df['Combined_Text'] = df['Application_Text'].fillna('') + " " + \
                          df['Customer_Feedback'].fillna('') + " " + \
                          df['Agent_Notes'].fillna('') + " " + \
                          df.get('Customer_Sentiment', pd.Series([''] * len(df))).fillna('')
    df = df.dropna(subset=['Application_Text']).reset_index(drop=True)
    return df

class SemanticSearchEngine:
    def __init__(self, model_name='all-mpnet-base-v2'):
        print(f"Loading SentenceTransformer model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.document_embeddings = None
        self.df = None
        
    def build_vector_index(self, df, text_column='Application_Text'):
        self.df = df
        documents = df[text_column].tolist()
        print(f"Encoding {len(documents)} documents to build vector index...")
        self.document_embeddings = self.model.encode(documents, show_progress_bar=True)
        print("Vector index built successfully.")
        
    def save_index(self, filepath):
        if self.document_embeddings is None or self.df is None:
            raise ValueError("Index or dataframe not built.")
        print(f"Saving vector index to {filepath}...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'embeddings': self.document_embeddings, 'df': self.df}, f)
        print("Index saved successfully.")
        
    def load_index(self, filepath):
        if os.path.exists(filepath):
            print(f"Loading vector index from {filepath}...")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.document_embeddings = data['embeddings']
                self.df = data['df']
            print("Index loaded successfully.")
            return True
        return False

    def search(self, query, top_k=5):
        if self.document_embeddings is None:
            raise ValueError("Vector index not built. Call build_vector_index() first.")
            
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'score': similarities[idx],
                'document': self.df.iloc[idx]['Combined_Text'],
                'customer_name': self.df.iloc[idx].get('Customer_Name', 'N/A'),
                'loan_status': self.df.iloc[idx].get('Loan_Status', 'N/A')
            })
        return results

if __name__ == "__main__":
    dataset_path = '../hdfc_loan_dataset_full_enriched_fixed_v2.csv'
    if not os.path.exists(dataset_path):
        dataset_path = '../../hdfc_loan_dataset_full_enriched_fixed_v2.csv'
        if not os.path.exists(dataset_path):
            dataset_path = 'hdfc_loan_dataset_full_enriched_fixed_v2.csv'
        
    print("Loading dataset...")
    df = load_data(dataset_path)
    
    # Initialize Semantic Search Engine
    search_engine = SemanticSearchEngine()
    
    # Check if index exists to load
    index_path = '../saved_models/semantic_index_v2.pkl'
    if not os.path.exists('../saved_models'):
        os.makedirs('../saved_models', exist_ok=True)
        
    if not search_engine.load_index(index_path):
        # Build vector index on Combined_Text
        search_engine.build_vector_index(df, text_column='Combined_Text')
        # Save index for production reuse
        search_engine.save_index(index_path)
    
    # Example queries from the assignment
    queries = [
        "Show customers requesting education loans.",
        "Find dissatisfied customers.",
        "Find applicants mentioning delayed approval.",
        "Customers requesting home renovation.",
        "Customers having positive experience with the branch."
    ]
    
    for q in queries:
        print(f"\n--- Query: '{q}' ---")
        top_results = search_engine.search(q, top_k=3)
        for i, res in enumerate(top_results):
            print(f"Rank {i+1} | Score: {res['score']:.4f} | Customer: {res['customer_name']} | Status: {res['loan_status']}")
            print(f"Document snippet: {res['document'][:150]}...\n")
