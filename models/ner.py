import spacy
import pandas as pd
from pathlib import Path

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
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / 'hdfc_loan_dataset_full_enriched_fixed_v2.csv'

        
    print("Loading dataset...")
    df = load_data(dataset_path)
    
    print("Loading spaCy model...")

    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")

    print("Extracting entities from Agent_Notes...")
    process_df = df.dropna(subset=['Agent_Notes']).copy()
    
    texts = process_df['Agent_Notes'].astype(str).tolist()
    process_df['Extracted_Entities'] = extract_entities_batch(texts, nlp)
    
    # Save the output to a CSV in data directory
    output_dir = repo_root / 'data'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'extracted_entities.csv'
    
    print(f"Saving extracted entities to {output_path}...")
    process_df[['Customer_Name', 'Bank', 'Loan_Amount', 'Agent_Notes', 'Extracted_Entities']].to_csv(output_path, index=False)
    
    print("NER processing completed successfully.")
