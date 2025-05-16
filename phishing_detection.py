import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Distribution of labels:\n{df['label'].value_counts()}")
    return df

# Feature extraction functions
def extract_url_features(url):
    features = {}
    
    # Parse URL
    parsed_url = urlparse(url)
    
    # Basic URL properties
    features['url_length'] = len(url)
    features['domain_length'] = len(parsed_url.netloc)
    features['path_length'] = len(parsed_url.path)
    
    # Number of subdomains
    features['subdomain_count'] = len(parsed_url.netloc.split('.')) - 1
    
    # Count special characters
    features['dots_count'] = url.count('.')
    features['hyphens_count'] = url.count('-')
    features['underscores_count'] = url.count('_')
    features['at_count'] = url.count('@')
    features['question_mark_count'] = url.count('?')
    features['ampersand_count'] = url.count('&')
    features['equals_count'] = url.count('=')
    
    # Presence of IP address in URL
    features['has_ip'] = 1 if bool(re.search(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc)) else 0
    
    # Presence of suspicious words
    suspicious_words = ['login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm', 'user', 
                         'password', 'bank', 'credit', 'free', 'lucky', 'bonus', 'prize']
    for word in suspicious_words:
        features[f'contains_{word}'] = 1 if word in url.lower() else 0
    
    # Protocol features
    features['is_https'] = 1 if parsed_url.scheme == 'https' else 0
    features['non_standard_port'] = 0
    if parsed_url.netloc.find(':') != -1:
        port = int(parsed_url.netloc.split(':')[1])
        features['non_standard_port'] = 1 if port != 80 and port != 443 else 0
    
    # URL path features
    features['path_depth'] = len([x for x in parsed_url.path.split('/') if x])
    features['has_multiple_subdomains'] = 1 if features['subdomain_count'] > 2 else 0
    
    # URLs with digits
    features['domain_has_digits'] = 1 if bool(re.search(r'\d', parsed_url.netloc)) else 0
    
    # TLD analysis
    tld = parsed_url.netloc.split('.')[-1] if len(parsed_url.netloc.split('.')) > 1 else ''
    features['tld_length'] = len(tld)
    
    # Unusual TLDs (more associated with phishing)
    unusual_tlds = ['xyz', 'top', 'club', 'online', 'site', 'info', 'icu', 'vip', 'click', 'buzz']
    features['unusual_tld'] = 1 if tld in unusual_tlds else 0
    
    # URL entropy (more randomness often means phishing)
    features['url_entropy'] = calculate_entropy(url)
    
    return features

def calculate_entropy(text):
    """Calculate Shannon entropy for a string"""
    text = text.lower()
    prob = [float(text.count(c)) / len(text) for c in set(text)]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

# Feature extraction pipeline
def extract_features_from_urls(urls):
    features_list = []
    for url in urls:
        features_list.append(extract_url_features(url))
    return pd.DataFrame(features_list)

# Text features from URL
def get_url_text_features(df):
    # Extract text features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, analyzer='char', ngram_range=(3, 5))
    X_text = vectorizer.fit_transform(df['url'])
    text_feature_names = [f'tfidf_{i}' for i in range(X_text.shape[1])]
    X_text_df = pd.DataFrame(X_text.toarray(), columns=text_feature_names)
    return X_text_df, vectorizer

# Main function
def main():
    # Load data
    df = load_data('urls.csv')
    
    # Explore data
    print("\nFirst few URLs:")
    for i, (url, label) in enumerate(zip(df['url'].head(), df['label'].head())):
        print(f"{i+1}. {url} - {label}")
    
    # Extract features
    print("\nExtracting features from URLs...")
    X_features = extract_features_from_urls(df['url'])
    
    # Get text features
    print("Extracting text features...")
    X_text, vectorizer = get_url_text_features(df)
    
    # Combine features
    X_combined = pd.concat([X_features, X_text], axis=1)
    y = (df['label'] == 'phishing').astype(int)  # Convert to binary
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
    print(f"Using {X_combined.shape[1]} features")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Phishing', 'Phishing']))
    
    # Feature importance
    print("\nTop 15 important features:")
    feature_importances = pd.DataFrame({
        'feature': X_combined.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importances.head(15))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Phishing', 'Phishing'], 
                yticklabels=['Not Phishing', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Save model and vectorizer for API use
    print("\nSaving model and vectorizer for API use...")
    joblib.dump(model, 'phishing_model.pkl')
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('X_columns.pkl', 'wb') as f:
        pickle.dump(X_combined.columns, f)
    print("Model and vectorizer saved successfully!")
    
    # Function to make predictions on new URLs
    def predict_phishing(urls, model, vectorizer):
        # Extract features
        features = extract_features_from_urls(urls)
        
        # Extract text features
        text_features = vectorizer.transform(urls)
        text_feature_names = [f'tfidf_{i}' for i in range(text_features.shape[1])]
        text_features_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names)
        
        # Combine features
        X = pd.concat([features, text_features_df], axis=1)
        
        # Adjust columns to match training data
        for col in X_combined.columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[X_combined.columns]  # Ensure same column order
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    # Example usage
    print("\nExample prediction:")
    test_urls = [
        "https://google.com",
        "http://amaz0n-secure-login.com",
        "https://facebook.com",
        "http://paypal-account-verify-secure.us"
    ]
    
    predictions, probabilities = predict_phishing(test_urls, model, vectorizer)
    for url, pred, prob in zip(test_urls, predictions, probabilities):
        print(f"URL: {url}")
        print(f"Prediction: {'Phishing' if pred == 1 else 'Not Phishing'}")
        print(f"Probability of being phishing: {prob:.4f}")
        print()

if __name__ == "__main__":
    main() 