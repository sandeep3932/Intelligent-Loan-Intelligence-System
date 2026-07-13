import pandas as pd
import numpy as np
import os
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, FastText
import gensim.downloader as api

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def main():
    print("=========================================")
    print("Starting Engineer 1: Task 4 (Embedding Models)")
    print("=========================================")

    # Load the real processed dataset
    dataset_path = 'data/processed_dataset.csv'
    if not os.path.exists(dataset_path):
        dataset_path = 'hdfc_loan_dataset_processed.csv'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Processed dataset not found at {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Input dataset shape: {df.shape}")

    # Processed Text Columns to form the corpus
    text_cols = ['Application_Text_processed', 'Customer_Feedback_processed', 'Agent_Notes_processed']
    
    # Combine text columns and tokenize
    corpus_text = df[text_cols].fillna('').agg(' '.join, axis=1).tolist()
    print("Tokenizing corpus...")
    corpus_tokens = [word_tokenize(sentence) for sentence in corpus_text]
    print(f"Total sentences in corpus: {len(corpus_tokens)}")
    
    # Flatten to get unique vocabulary size for comparison
    vocab = set(word for sentence in corpus_tokens for word in sentence)
    
    os.makedirs('saved_models', exist_ok=True)
    models = {}

    # ========================================
    # Task 4.1: Word2Vec - CBOW
    # ========================================
    print("\n--- Task 4.1: Word2Vec - CBOW ---")
    w2v_cbow = Word2Vec(sentences=corpus_tokens, vector_size=100, window=5, min_count=1, sg=0)
    w2v_cbow.save('saved_models/w2v_cbow.model')
    models['Word2Vec (CBOW)'] = w2v_cbow
    print("Word2Vec CBOW trained and saved to saved_models/w2v_cbow.model")

    # ========================================
    # Task 4.2: Word2Vec - Skip-Gram
    # ========================================
    print("\n--- Task 4.2: Word2Vec - Skip-Gram ---")
    w2v_sg = Word2Vec(sentences=corpus_tokens, vector_size=100, window=5, min_count=1, sg=1)
    w2v_sg.save('saved_models/w2v_sg.model')
    models['Word2Vec (Skip-Gram)'] = w2v_sg
    print("Word2Vec Skip-Gram trained and saved to saved_models/w2v_sg.model")

    # ========================================
    # Task 4.3: FastText
    # ========================================
    print("\n--- Task 4.3: FastText ---")
    fasttext_model = FastText(sentences=corpus_tokens, vector_size=100, window=5, min_count=1)
    fasttext_model.save('saved_models/fasttext.model')
    models['FastText'] = fasttext_model
    print("FastText trained and saved to saved_models/fasttext.model")

    # ========================================
    # Task 4.4: GloVe
    # ========================================
    print("\n--- Task 4.4: GloVe ---")
    print("Loading pre-trained GloVe model (glove-wiki-gigaword-50)... this may take a moment.")
    glove_model = api.load("glove-wiki-gigaword-50")
    models['GloVe'] = glove_model
    print("GloVe model loaded.")

    # ========================================
    # Task 4.5: Compare Vocabulary Coverage, OOV Handling and Similarity Performance
    # ========================================
    print("\n--- Task 4.5: Comparison Results ---")
    
    test_word = "loan"
    oov_word = "unbelievablexyz"

    for name, model in models.items():
        print(f"\nModel: {name}")
        
        # 1. Vocabulary Coverage
        if hasattr(model, 'key_to_index'): # pretrained glove
            model_vocab = set(model.key_to_index.keys())
        else: # gensim models
            model_vocab = set(model.wv.key_to_index.keys())
            
        in_vocab = len(vocab.intersection(model_vocab))
        coverage = (in_vocab / len(vocab)) * 100 if len(vocab) > 0 else 0
        print(f"  - Vocabulary Coverage: {coverage:.2f}% ({in_vocab}/{len(vocab)} words)")
        
        # 2. OOV Handling
        print(f"  - OOV Handling (Testing word: '{oov_word}'):")
        try:
            if hasattr(model, 'wv'):
                _ = model.wv[oov_word]
            else:
                _ = model[oov_word]
            print(f"      Success: Generated subword embeddings.")
        except KeyError:
            print(f"      Failed: KeyError (Word not in vocabulary).")
            
        # 3. Similarity Performance
        print(f"  - Similarity Performance (Testing word: '{test_word}'):")
        try:
            # check if the word is in the model's vocabulary
            vocab_dict = model.wv.key_to_index if hasattr(model, 'wv') else model.key_to_index
            if test_word in vocab_dict:
                similars = model.wv.most_similar(test_word, topn=3) if hasattr(model, 'wv') else model.most_similar(test_word, topn=3)
                for w, score in similars:
                    print(f"      {w}: {score:.4f}")
            else:
                print(f"      '{test_word}' not found in model vocabulary.")
        except Exception as e:
            print(f"      Error calculating similarity: {e}")

    print("\n=========================================")
    print("Task 4 Completed Successfully.")
    print("=========================================")

if __name__ == "__main__":
    main()
