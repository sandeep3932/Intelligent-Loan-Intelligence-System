import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    # Ensure no missing values in text or target
    df = df.dropna(subset=['Customer_Feedback', 'Customer_Sentiment'])
    return df

def build_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Linear SVM': LinearSVC(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # for ROC-AUC need probabilities or decision function
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            # handle multi-class roc auc if needed, assuming binary for now
            if len(np.unique(y_train)) > 2:
                roc_auc = roc_auc_score(y_test, probs, multi_class='ovr')
            else:
                roc_auc = roc_auc_score(y_test, probs[:, 1])
        else:
            decision = model.decision_function(X_test)
            if len(np.unique(y_train)) > 2:
                # Approximate or use ovr, for LinearSVC predict_proba is not available
                # LinearSVC decision function can be used but it's more complex for multi-class
                roc_auc = np.nan # placeholder if multi-class
            else:
                roc_auc = roc_auc_score(y_test, decision)
                
        # If multi-class, need average='macro' or 'weighted'
        if len(np.unique(y_train)) > 2:
            avg = 'weighted'
        else:
            avg = 'binary'
            
        precision = precision_score(y_test, preds, average=avg)
        recall = recall_score(y_test, preds, average=avg)
        f1 = f1_score(y_test, preds, average=avg)
        
        results[name] = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC-AUC': roc_auc
        }
    return results

if __name__ == "__main__":
    dataset_path = '../../hdfc_loan_dataset_full_enriched.csv'
    if not os.path.exists(dataset_path):
        dataset_path = 'hdfc_loan_dataset_full_enriched.csv'
    
    print(f"Loading data from {dataset_path}...")
    df = load_data(dataset_path)
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['Customer_Sentiment'])
    print(f"Classes found: {le.classes_}")
    
    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['Customer_Feedback'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training and evaluating models...")
    results = build_models(X_train, X_test, y_train, y_test)
    
    print("\n--- Model Evaluation ---")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}" if not np.isnan(value) else f"  {metric_name}: N/A")
