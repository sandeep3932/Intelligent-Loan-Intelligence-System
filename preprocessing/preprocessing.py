import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

class TextPreprocessor:
    """
    Class to handle all text preprocessing and EDA tasks.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    # ============================
    # Task 1: Text Preprocessing
    # ============================
    def preprocess_text(self, text, use_stemming=False):
        """
        Applies a full preprocessing pipeline to a single text string.
        """
        if not isinstance(text, str):
            return ""

        # Lowercasing: Normalize text to lower case to reduce vocabulary size
        text = text.lower()
        
        # Regex cleaning & Punctuation removal & Number removal: 
        # Remove non-alphabet characters to keep only words. This removes numbers and punctuation.
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Tokenization: Split text into individual words
        tokens = word_tokenize(text)
        
        # Stop word removal: Remove common words that don't add much meaning
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatization or Stemming: Reduce words to their root form
        if use_stemming:
            # Stemming is a faster, rule-based approach to get the root
            processed_tokens = [self.stemmer.stem(word) for word in tokens]
        else:
            # Lemmatization is dictionary-based and yields actual words
            processed_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
        return ' '.join(processed_tokens)

    def process_dataframe(self, df, columns, use_stemming=False):
        """
        Processes a list of text columns in a DataFrame, creating new processed columns.
        Preserves original columns as requested.
        """
        df_processed = df.copy()
        for col in columns:
            if col in df_processed.columns:
                print(f"Processing column: {col}")
                # Create a new column with '_processed' suffix to preserve the original
                df_processed[f'{col}_processed'] = df_processed[col].apply(
                    lambda x: self.preprocess_text(x, use_stemming=use_stemming)
                )
        return df_processed

    # ============================
    # Task 2: Exploratory NLP Analysis
    # ============================
    def get_corpus(self, df, column):
        """Helper to combine all text in a column into a single string/list of tokens."""
        text = ' '.join(df[column].dropna().astype(str).tolist())
        return word_tokenize(text)

    def plot_word_frequency(self, df, column, top_n=20):
        """
        Word Frequency: Finds the most common words to understand primary topics.
        """
        tokens = self.get_corpus(df, column)
        counter = Counter(tokens)
        common_words = counter.most_common(top_n)
        
        if not common_words:
            return common_words
            
        words, counts = zip(*common_words)
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts)
        plt.title(f'Top {top_n} Words in {column}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'data/{column}_word_freq.png')
        plt.close()
        return common_words

    def get_ngrams(self, df, column, n=2, top_k=10):
        """
        Bigrams/Trigrams: Extracts multi-word phrases for deeper contextual understanding.
        """
        from nltk.util import ngrams
        tokens = self.get_corpus(df, column)
        n_grams = ngrams(tokens, n)
        counter = Counter(n_grams)
        return counter.most_common(top_k)

    def generate_word_cloud(self, df, column):
        """
        Word Clouds: Visual representation of word frequencies.
        """
        text = ' '.join(df[column].dropna().astype(str).tolist())
        if not text.strip():
            return
            
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {column}')
        plt.tight_layout()
        plt.savefig(f'data/{column}_wordcloud.png')
        plt.close()

    def vocabulary_statistics(self, df, column):
        """
        Vocabulary Statistics: Computes total words and unique words (lexical richness).
        """
        tokens = self.get_corpus(df, column)
        total_words = len(tokens)
        unique_words = len(set(tokens))
        print(f"--- Vocab Stats for {column} ---")
        print(f"Total Words: {total_words}")
        print(f"Unique Words (Vocabulary Size): {unique_words}")
        print(f"Lexical Richness (Unique/Total): {unique_words/total_words:.4f}" if total_words > 0 else "0")
        return total_words, unique_words


if __name__ == "__main__":
    print("=========================================")
    print("Starting Preprocessing & EDA Pipeline")
    print("=========================================")
    
    dataset_path = 'hdfc_loan_dataset_full_enriched.csv'
    if not os.path.exists(dataset_path):
        dataset_path = '../hdfc_loan_dataset_full_enriched.csv' # Try root if running from preprocessing/
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Error: Real dataset not found at {dataset_path}. Please place hdfc_loan_dataset_full_enriched.csv in the directory.")

    print(f"Input file used: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Input dataset shape: {df.shape}")

    text_cols = ['Application_Text', 'Customer_Feedback', 'Agent_Notes']
    
    # Ensure columns exist
    for col in text_cols:
        if col not in df.columns:
            raise ValueError(f"Required text column {col} is missing from the dataset.")

    preprocessor = TextPreprocessor()
    
    # Task 1 execution
    print("\n--- Task 1: Text Preprocessing ---")
    df_processed = preprocessor.process_dataframe(df, text_cols)
    print("Text preprocessing completed. New columns appended.")
    
    # Task 2 execution
    print("\n--- Task 2: Word Frequency Analysis & EDA ---")
    os.makedirs('data', exist_ok=True)
    for col in text_cols:
        processed_col = f"{col}_processed"
        if processed_col in df_processed.columns:
            preprocessor.plot_word_frequency(df_processed, processed_col)
            print(f"\nTop Bigrams for {col}:", preprocessor.get_ngrams(df_processed, processed_col, n=2, top_k=5))
            print(f"Top Trigrams for {col}:", preprocessor.get_ngrams(df_processed, processed_col, n=3, top_k=5))
            preprocessor.generate_word_cloud(df_processed, processed_col)
            preprocessor.vocabulary_statistics(df_processed, processed_col)

    # Save the processed dataframe
    output_path = 'data/processed_dataset.csv'
    df_processed.to_csv(output_path, index=False)
    
    print("\n=========================================")
    print("Summary:")
    print("Tasks completed: Task 1 (Text Preprocessing), Task 2 (Exploratory NLP Analysis)")
    print(f"Original dataset rows: {len(df)} | Processed dataset rows: {len(df_processed)}")
    print(f"Output files generated: {output_path} and EDA plots in data/ directory")
    print("=========================================")
