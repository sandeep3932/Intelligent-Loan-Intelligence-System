import spacy
import pandas as pd
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def extract_entities_batch(texts, nlp):
    # Process texts in batches for better performance using nlp.pipe
    # Entities to be recognized: person, organization, location, date and money

    allowed_labels = {'PERSON', 'ORG', 'LOC', 'GPE', 'DATE', 'MONEY'}
    all_entities = []
    

    for doc in nlp.pipe(texts, batch_size=50):
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in allowed_labels]
        all_entities.append(entities)
        
    return all_entities

if __name__ == "__main__":
    # NER requires the original unprocessed text (capitalization, punctuation intact)
    dataset_path = 'hdfc_loan_dataset_full_enriched.csv'

        
    print("Loading dataset...")
    df = load_data(dataset_path)
    
    print("Loading spaCy model...")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    print("Extracting entities from a sample of Agent_Notes...")
    sample_df = df.dropna(subset=['Agent_Notes']).head(20).copy()
    
    texts = sample_df['Agent_Notes'].astype(str).tolist()
    sample_df['Extracted_Entities'] = extract_entities_batch(texts, nlp)
    
    for idx, row in sample_df.iterrows():
        print(f"\nRow {idx}")
        print(f"Agent_Notes: {row['Agent_Notes']}")
        print(f"Entities: {row['Extracted_Entities']}")
        print(f"Structured Data - Customer: {row['Customer_Name']}, Bank: {row['Bank']}, Amount: {row['Loan_Amount']}")
