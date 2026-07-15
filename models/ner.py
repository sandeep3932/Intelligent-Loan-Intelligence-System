import spacy
import spacy.cli
import pandas as pd
from pathlib import Path

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def load_ner_model():
    """Load the custom-trained NER model, falling back to en_core_web_md."""
    repo_root = Path(__file__).resolve().parent.parent
    custom_model_path = repo_root / 'saved_models' / 'custom_ner'
    
    if custom_model_path.exists():
        try:
            nlp = spacy.load(str(custom_model_path))
            print(f"Loaded custom NER model from {custom_model_path}")
            return nlp
        except Exception as e:
            print(f"Failed to load custom model: {e}. Falling back to en_core_web_md.")
    
    try:
        nlp = spacy.load("en_core_web_md")
        print("Loaded fallback model: en_core_web_md")
        return nlp
    except OSError:
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")
        print("Downloaded and loaded fallback model: en_core_web_md")
        return nlp

def extract_entities_batch(texts, nlp):
    # Process texts in batches for better performance using nlp.pipe
    # Entities to be recognized: person, organization, location, date and money

    allowed_labels = {'PERSON', 'ORG', 'LOC', 'MONEY', 'GPE', 'DATE'}
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
    
    print("Loading NER model...")
    nlp = load_ner_model()

    # Use Application_Text (rich in entities: names, banks, cities, loan amounts)
    # instead of Agent_Notes (which are generic templates with no real entities)
    print("Extracting entities from Application_Text...")
    process_df = df.dropna(subset=['Application_Text']).copy()
    
    texts = process_df['Application_Text'].astype(str).tolist()
    process_df['Extracted_Entities'] = extract_entities_batch(texts, nlp)
    
    # Save the output to a CSV in data directory
    output_dir = repo_root / 'data'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'extracted_entities.csv'
    
    print(f"Saving extracted entities to {output_path}...")
    process_df[['Customer_Name', 'Bank', 'Loan_Amount', 'Application_Text', 'Extracted_Entities']].to_csv(output_path, index=False)
    
    print("NER processing completed successfully.")

