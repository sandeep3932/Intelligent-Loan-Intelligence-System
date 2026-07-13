import spacy
import pandas as pd
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def extract_entities(text, nlp):
    if not isinstance(text, str):
        return []
    doc = nlp(text)
    # Extract Person, Organization, Location, Dates, Money
    # spaCy tags: PERSON, ORG, LOC, GPE, DATE, MONEY
    allowed_labels = {'PERSON', 'ORG', 'LOC', 'GPE', 'DATE', 'MONEY'}
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in allowed_labels]
    return entities

if __name__ == "__main__":
    dataset_path = '../hdfc_loan_dataset_full_enriched.csv'
    if not os.path.exists(dataset_path):
        dataset_path = 'hdfc_loan_dataset_full_enriched.csv'
        
    print("Loading dataset...")
    df = load_data(dataset_path)
    
    print("Loading spaCy model...")
    # Make sure to run: python -m spacy download en_core_web_sm
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    print("Extracting entities from a sample of Agent_Notes...")
    sample_df = df.dropna(subset=['Agent_Notes']).head(10).copy()
    
    sample_df['Extracted_Entities'] = sample_df['Agent_Notes'].apply(lambda x: extract_entities(x, nlp))
    
    for idx, row in sample_df.iterrows():
        print(f"\nRow {idx}")
        print(f"Agent_Notes: {row['Agent_Notes']}")
        print(f"Entities: {row['Extracted_Entities']}")
        print(f"Structured Data - Customer: {row['Customer_Name']}, Bank: {row['Bank']}, Amount: {row['Loan_Amount']}")
