"""
NER Model Training Script
==========================
Fine-tunes a spaCy NER model on loan-domain text using automatic annotation
from structured dataset columns (Customer_Name, Bank, City, State, Loan_Amount).

Pipeline:
1. Load dataset and generate entity annotations from structured columns
2. Split into Train (70%) / Validation (15%) / Test (15%)
3. Convert to spaCy DocBin format
4. Train using spaCy's training API
5. Validate against quality rules (F1 >= 0.70, per-type recall >= 0.60, precision >= 0.50)
6. Save the trained model to saved_models/custom_ner/
"""

import spacy
import spacy.cli
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
import pandas as pd
import numpy as np
import re
import random
import json
from pathlib import Path
from collections import defaultdict


RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Validation thresholds
MIN_OVERALL_F1 = 0.70
MIN_PER_TYPE_RECALL = 0.60
MIN_PER_TYPE_PRECISION = 0.50
MIN_COVERAGE = 0.80  # At least 80% of texts should have entities detected

# Training hyperparameters
N_ITER = 40
DROPOUT = 0.3
BATCH_SIZE_START = 4
BATCH_SIZE_END = 32
PATIENCE = 5  # Early stopping patience (consecutive validations without improvement)

# Entity labels we care about
ENTITY_LABELS = ['PERSON', 'ORG', 'GPE', 'MONEY']

# Repo paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / 'hdfc_loan_dataset_full_enriched_fixed_v2.csv'
OUTPUT_MODEL_DIR = REPO_ROOT / 'saved_models' / 'custom_ner'
DATA_DIR = REPO_ROOT / 'data' / 'ner_training'




def find_entity_spans(text, entity_text, label):
    """Find all non-overlapping occurrences of entity_text in text and return span tuples."""
    spans = []
    if not entity_text or not text:
        return spans
    
    # Escape for regex but do exact matching
    pattern = re.escape(str(entity_text))
    for match in re.finditer(pattern, text):
        spans.append((match.start(), match.end(), label))
    return spans


def find_money_spans(text):
    """Find INR money patterns in text like 'INR 8,031,545' or 'INR 198,671'."""
    spans = []
    # Pattern: INR followed by digits with optional commas
    pattern = r'INR\s+[\d,]+'
    for match in re.finditer(pattern, text):
        spans.append((match.start(), match.end(), 'MONEY'))
    return spans


def remove_overlapping_spans(spans):
    """Remove overlapping spans, keeping the longest one."""
    if not spans:
        return spans
    
    # Sort by start position, then by length (descending)
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    
    filtered = [spans[0]]
    for span in spans[1:]:
        last = filtered[-1]
        # If this span doesn't overlap with the last accepted span
        if span[0] >= last[1]:
            filtered.append(span)
    
    return filtered


def generate_annotations(df):
    """
    Generate NER annotations from structured dataset columns.
    Uses Application_Text as the primary text source and maps:
      - Customer_Name -> PERSON
      - Bank -> ORG
      - City, State -> GPE
      - INR patterns -> MONEY
    """
    annotations = []
    skipped = 0
    
    for idx, row in df.iterrows():
        text = str(row.get('Application_Text', ''))
        if not text or text == 'nan':
            skipped += 1
            continue
        
        spans = []
        
        # PERSON: Customer_Name
        name = str(row.get('Customer_Name', ''))
        if name and name != 'nan':
            spans.extend(find_entity_spans(text, name, 'PERSON'))
        
        # ORG: Bank
        bank = str(row.get('Bank', ''))
        if bank and bank != 'nan':
            spans.extend(find_entity_spans(text, bank, 'ORG'))
        
        # GPE: City
        city = str(row.get('City', ''))
        if city and city != 'nan':
            spans.extend(find_entity_spans(text, city, 'GPE'))
        
        # GPE: State
        state = str(row.get('State', ''))
        if state and state != 'nan':
            spans.extend(find_entity_spans(text, state, 'GPE'))
        
        # MONEY: INR patterns
        spans.extend(find_money_spans(text))
        
        # Remove overlaps
        spans = remove_overlapping_spans(spans)
        
        if spans:
            annotations.append((text, {"entities": spans}))
    
    print(f"Generated {len(annotations)} annotated examples ({skipped} skipped due to missing text)")
    return annotations



# TRAIN / VALIDATION / TEST SPLIT (70-15-15)


def split_data(annotations, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO):
    """Split annotations into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(RANDOM_SEED)
    data = list(annotations)
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data



def annotations_to_docbin(annotations, nlp):
    """Convert annotation tuples to spaCy DocBin."""
    db = DocBin()
    success = 0
    failed = 0
    
    for text, annot in annotations:
        doc = nlp.make_doc(text)
        ents = []
        spans = annot["entities"]
        
        for start, end, label in spans:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        
        # Filter out overlapping spans using spaCy's built-in filter
        try:
            doc.ents = ents
            db.add(doc)
            success += 1
        except ValueError:
            # If there are overlapping spans, try to resolve
            from spacy.util import filter_spans
            doc.ents = filter_spans(ents)
            db.add(doc)
            success += 1
    
    print(f"  DocBin: {success} docs added, {failed} failed")
    return db


def save_docbins(train_data, val_data, test_data, nlp):
    """Save train/val/test DocBins to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Converting train data...")
    train_db = annotations_to_docbin(train_data, nlp)
    train_db.to_disk(DATA_DIR / "train.spacy")
    
    print("Converting val data...")
    val_db = annotations_to_docbin(val_data, nlp)
    val_db.to_disk(DATA_DIR / "dev.spacy")
    
    print("Converting test data...")
    test_db = annotations_to_docbin(test_data, nlp)
    test_db.to_disk(DATA_DIR / "test.spacy")
    
    print(f"Saved DocBins to {DATA_DIR}")



def create_training_examples(data, nlp):
    """Convert annotation data to spaCy Example objects."""
    examples = []
    for text, annot in data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annot)
        examples.append(example)
    return examples


def train_ner_model(train_data, val_data, nlp):
    """
    Fine-tune the NER component of the spaCy model.
    Uses early stopping based on validation F1.
    """
    # Get the NER pipe (or create if not exists)
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Add entity labels
    for label in ENTITY_LABELS:
        ner.add_label(label)
    
    # Prepare training examples
    train_examples = create_training_examples(train_data, nlp)
    val_examples = create_training_examples(val_data, nlp)
    
    # Only train the NER component
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    best_f1 = 0.0
    patience_counter = 0
    best_model_bytes = None
    training_log = []
    
    print(f"\nStarting training for {N_ITER} iterations...")
    print(f"Early stopping patience: {PATIENCE} iterations")
    print("-" * 70)
    
    with nlp.disable_pipes(*other_pipes):
        # Reset and initialize optimizer
        optimizer = nlp.resume_training()
        
        batch_sizes = compounding(BATCH_SIZE_START, BATCH_SIZE_END, 1.001)
        
        for iteration in range(1, N_ITER + 1):
            random.shuffle(train_examples)
            losses = {}
            
            batches = minibatch(train_examples, size=batch_sizes)
            for batch in batches:
                nlp.update(batch, drop=DROPOUT, sgd=optimizer, losses=losses)
            
            # Evaluate on validation set
            val_scores = evaluate_model(nlp, val_examples)
            val_f1 = val_scores.get('ents_f', 0.0)
            val_p = val_scores.get('ents_p', 0.0)
            val_r = val_scores.get('ents_r', 0.0)
            
            log_entry = {
                'iteration': iteration,
                'train_loss': losses.get('ner', 0.0),
                'val_f1': val_f1,
                'val_precision': val_p,
                'val_recall': val_r
            }
            training_log.append(log_entry)
            
            print(f"Iter {iteration:3d} | Loss: {losses.get('ner', 0.0):8.2f} | "
                  f"Val F1: {val_f1:.4f} | P: {val_p:.4f} | R: {val_r:.4f}")
            
            # Early stopping check
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_bytes = nlp.to_bytes()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\nEarly stopping at iteration {iteration} (no improvement for {PATIENCE} iterations)")
                    break
    
    # Restore best model
    if best_model_bytes:
        nlp.from_bytes(best_model_bytes)
        print(f"\nRestored best model with Val F1: {best_f1:.4f}")
    
    return nlp, training_log


def evaluate_model(nlp, examples):
    """Evaluate model on a set of Examples and return scores."""
    scorer = Scorer()
    
    # Create predicted examples
    pred_examples = []
    for example in examples:
        pred_doc = nlp(example.reference.text)
        pred_example = Example(pred_doc, example.reference)
        pred_examples.append(pred_example)
    
    scores = scorer.score(pred_examples)
    return scores


# VALIDATION RULES


def evaluate_per_entity_type(nlp, test_data):
    """Evaluate model performance per entity type."""
    type_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for text, annot in test_data:
        doc = nlp(text)
        
        # Gold entities
        gold_entities = set()
        for start, end, label in annot['entities']:
            gold_entities.add((start, end, label))
        
        # Predicted entities
        pred_entities = set()
        for ent in doc.ents:
            if ent.label_ in ENTITY_LABELS:
                pred_entities.add((ent.start_char, ent.end_char, ent.label_))
        
        # Calculate TP, FP, FN per type
        for label in ENTITY_LABELS:
            gold_of_type = {(s, e) for s, e, l in gold_entities if l == label}
            pred_of_type = {(s, e) for s, e, l in pred_entities if l == label}
            
            tp = len(gold_of_type & pred_of_type)
            fp = len(pred_of_type - gold_of_type)
            fn = len(gold_of_type - pred_of_type)
            
            type_stats[label]['tp'] += tp
            type_stats[label]['fp'] += fp
            type_stats[label]['fn'] += fn
    
    results = {}
    for label in ENTITY_LABELS:
        stats = type_stats[label]
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return results


def evaluate_coverage(nlp, test_data):
    """Check what percentage of test texts have at least one entity detected."""
    texts_with_entities = 0
    total = len(test_data)
    
    for text, _ in test_data:
        doc = nlp(text)
        if any(ent.label_ in ENTITY_LABELS for ent in doc.ents):
            texts_with_entities += 1
    
    coverage = texts_with_entities / total if total > 0 else 0.0
    return coverage


def validate_model(nlp, test_data):
    """
    Run all validation rules against the trained model.
    Returns (passed: bool, report: dict)
    """
    print("\n" + "=" * 70)
    print("VALIDATION RULES CHECK")
    print("=" * 70)
    
    # Create test examples for overall scoring
    test_examples = create_training_examples(test_data, nlp)
    overall_scores = evaluate_model(nlp, test_examples)
    
    overall_f1 = overall_scores.get('ents_f', 0.0)
    overall_p = overall_scores.get('ents_p', 0.0)
    overall_r = overall_scores.get('ents_r', 0.0)
    
    # Per-entity-type evaluation
    per_type = evaluate_per_entity_type(nlp, test_data)
    
    # Coverage evaluation
    coverage = evaluate_coverage(nlp, test_data)
    
    # Print results
    print(f"\nOverall: F1={overall_f1:.4f} | P={overall_p:.4f} | R={overall_r:.4f}")
    print(f"Coverage: {coverage:.2%} of texts have entities detected")
    print(f"\nPer-entity-type results:")
    print(f"{'Type':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 60)
    
    all_passed = True
    failures = []
    
    for label in ENTITY_LABELS:
        stats = per_type[label]
        p, r, f1 = stats['precision'], stats['recall'], stats['f1']
        print(f"{label:<10} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {stats['tp']:>5} {stats['fp']:>5} {stats['fn']:>5}")
    
    # Rule 1: Overall F1 >= 0.70
    print(f"\n--- Rule Checks ---")
    rule1 = overall_f1 >= MIN_OVERALL_F1
    print(f"[{'PASS' if rule1 else 'FAIL'}] Overall F1 >= {MIN_OVERALL_F1}: {overall_f1:.4f}")
    if not rule1:
        all_passed = False
        failures.append(f"Overall F1 {overall_f1:.4f} < {MIN_OVERALL_F1}")
    
    # Rule 2: Per-type recall >= 0.60
    for label in ENTITY_LABELS:
        recall = per_type[label]['recall']
        passed = recall >= MIN_PER_TYPE_RECALL
        print(f"[{'PASS' if passed else 'FAIL'}] {label} Recall >= {MIN_PER_TYPE_RECALL}: {recall:.4f}")
        if not passed:
            all_passed = False
            failures.append(f"{label} recall {recall:.4f} < {MIN_PER_TYPE_RECALL}")
    
    # Rule 3: Per-type precision >= 0.50
    for label in ENTITY_LABELS:
        precision = per_type[label]['precision']
        passed = precision >= MIN_PER_TYPE_PRECISION
        print(f"[{'PASS' if passed else 'FAIL'}] {label} Precision >= {MIN_PER_TYPE_PRECISION}: {precision:.4f}")
        if not passed:
            all_passed = False
            failures.append(f"{label} precision {precision:.4f} < {MIN_PER_TYPE_PRECISION}")
    
    # Rule 4: Coverage >= 80%
    rule4 = coverage >= MIN_COVERAGE
    print(f"[{'PASS' if rule4 else 'FAIL'}] Coverage >= {MIN_COVERAGE:.0%}: {coverage:.2%}")
    if not rule4:
        all_passed = False
        failures.append(f"Coverage {coverage:.2%} < {MIN_COVERAGE:.0%}")
    
    print(f"\n{'ALL VALIDATION RULES PASSED!' if all_passed else 'SOME VALIDATION RULES FAILED!'}")
    if failures:
        for f in failures:
            print(f"  - {f}")
    
    report = {
        'overall_f1': overall_f1,
        'overall_precision': overall_p,
        'overall_recall': overall_r,
        'per_type': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv 
                         for kk, vv in v.items()} for k, v in per_type.items()},
        'coverage': coverage,
        'all_passed': all_passed,
        'failures': failures
    }
    
    return all_passed, report



def save_model(nlp, report, training_log):
    """Save the trained model and training report."""
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the spaCy model
    nlp.to_disk(OUTPUT_MODEL_DIR)
    print(f"\nModel saved to {OUTPUT_MODEL_DIR}")
    
    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # Save training report
    report_path = OUTPUT_MODEL_DIR / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'validation_report': report,
            'training_log': training_log,
            'config': {
                'train_ratio': TRAIN_RATIO,
                'val_ratio': VAL_RATIO,
                'test_ratio': TEST_RATIO,
                'n_iter': N_ITER,
                'dropout': DROPOUT,
                'entity_labels': ENTITY_LABELS,
                'random_seed': RANDOM_SEED
            }
        }, f, indent=2, cls=NumpyEncoder)
    print(f"Training report saved to {report_path}")



# SMOKE TEST


def smoke_test(nlp):
    """Quick test on sample texts to verify entities are detected."""
    print("\n" + "=" * 70)
    print("SMOKE TEST")
    print("=" * 70)
    
    test_texts = [
        "Rohan Verma approached HDFC Bank seeking a home loan of INR 8,031,545.",
        "A personal loan application for INR 658,856 has been filed by Ananya Joshi of Vadodara.",
        "This is a business loan request from Harpreet Singh, currently residing in Kolkata, for INR 2,500,000.",
        "Priya Sharma submitted an education loan application to HDFC Bank for INR 1,200,000 from Chennai.",
        "HDFC Bank received an auto loan request from Rajesh Kumar of Mumbai for INR 750,000.",
    ]
    
    for text in test_texts:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ENTITY_LABELS]
        print(f"\nText: {text[:80]}...")
        print(f"  Entities: {entities}")



def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print("=" * 70)
    print("NER MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")
    
    # Step 1: Generate annotations
    print("\n--- Step 1: Generating annotations ---")
    annotations = generate_annotations(df)
    
    # Print annotation statistics
    entity_counts = defaultdict(int)
    for _, annot in annotations:
        for _, _, label in annot['entities']:
            entity_counts[label] += 1
    print(f"Entity counts: {dict(entity_counts)}")
    
    # Step 2: Split data
    print("\n--- Step 2: Splitting data (70/15/15) ---")
    train_data, val_data, test_data = split_data(annotations)
    
    # Step 3: Load base model and prepare data
    print("\n--- Step 3: Loading base spaCy model ---")
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        print("Downloading en_core_web_md...")
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")
    
    print(f"Base model pipeline: {nlp.pipe_names}")
    
    # Save DocBins for reference
    print("\n--- Step 3b: Saving DocBins ---")
    save_docbins(train_data, val_data, test_data, nlp)
    
    # Step 4: Train
    print("\n--- Step 4: Training NER model ---")
    nlp, training_log = train_ner_model(train_data, val_data, nlp)
    
    # Step 5: Validate
    print("\n--- Step 5: Validating model ---")
    passed, report = validate_model(nlp, test_data)
    
    # Step 6: Save (even if validation partially fails, save for iteration)
    print("\n--- Step 6: Saving model ---")
    save_model(nlp, report, training_log)
    
    # Smoke test
    smoke_test(nlp)
    
    if not passed:
        print("\n[WARNING] Some validation rules did not pass. Review the report and consider:")
        print("  - Increasing N_ITER")
        print("  - Adjusting DROPOUT")
        print("  - Adding more training data or improving annotations")
    else:
        print("\n[SUCCESS] All validation rules passed! Model is ready for deployment.")
    
    return passed


if __name__ == "__main__":
    success = main()
