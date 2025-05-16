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

class PhishingDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_columns = None
    
    def extract_url_features(self, url):
        features = {}
        
        
        parsed_url = urlparse(url)
        
        # Original length features - keeping them but adding improved versions
        features['url_length'] = len(url)
        features['domain_length'] = len(parsed_url.netloc)
        features['path_length'] = len(parsed_url.path)
        
        # Normalized length features
        features['url_length_normalized'] = len(url) / 100  # Normalize to reduce impact of extreme values
        features['domain_length_normalized'] = len(parsed_url.netloc) / 50
        features['path_length_normalized'] = len(parsed_url.path) / 50
        
        # Ratio features
        features['path_to_url_ratio'] = len(parsed_url.path) / len(url) if len(url) > 0 else 0
        features['domain_to_url_ratio'] = len(parsed_url.netloc) / len(url) if len(url) > 0 else 0
        
        # Threshold features for extreme values
        features['extremely_long_url'] = 1 if len(url) > 200 else 0
        features['extremely_long_domain'] = 1 if len(parsed_url.netloc) > 50 else 0
        features['extremely_long_path'] = 1 if len(parsed_url.path) > 100 else 0
        
        # Original features continue below
        features['subdomain_count'] = len(parsed_url.netloc.split('.')) - 1
        
        
        features['dots_count'] = url.count('.')
        features['hyphens_count'] = url.count('-')
        features['underscores_count'] = url.count('_')
        features['at_count'] = url.count('@')
        features['question_mark_count'] = url.count('?')
        features['ampersand_count'] = url.count('&')
        features['equals_count'] = url.count('=')
        
        
        features['has_ip'] = 1 if bool(re.search(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc)) else 0
        
        
        suspicious_words = ['login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm', 'user', 
                         'password', 'bank', 'credit', 'free', 'lucky', 'bonus', 'prize']
        for word in suspicious_words:
            features[f'contains_{word}'] = 1 if word in url.lower() else 0
        
        
        features['is_https'] = 1 if parsed_url.scheme == 'https' else 0
        features['non_standard_port'] = 0
        if parsed_url.netloc.find(':') != -1:
            port = int(parsed_url.netloc.split(':')[1])
            features['non_standard_port'] = 1 if port != 80 and port != 443 else 0
        
        
        features['path_depth'] = len([x for x in parsed_url.path.split('/') if x])
        features['has_multiple_subdomains'] = 1 if features['subdomain_count'] > 2 else 0
        
        
        features['domain_has_digits'] = 1 if bool(re.search(r'\d', parsed_url.netloc)) else 0
        
        
        tld = parsed_url.netloc.split('.')[-1] if len(parsed_url.netloc.split('.')) > 1 else ''
        features['tld_length'] = len(tld)
        
        
        unusual_tlds = ['xyz', 'top', 'club', 'online', 'site', 'info', 'icu', 'vip', 'click', 'buzz']
        features['unusual_tld'] = 1 if tld in unusual_tlds else 0
        
        
        features['url_entropy'] = self.calculate_entropy(url)
        
        return features

    def calculate_entropy(self, text):
        """Calculate Shannon entropy for a string"""
        text = text.lower()
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        entropy = -sum([p * np.log2(p) for p in prob])
        return entropy

    def extract_features_from_urls(self, urls):
        features_list = []
        for url in urls:
            features_list.append(self.extract_url_features(url))
        return pd.DataFrame(features_list)

    def fit(self, urls, labels):
        
        X_features = self.extract_features_from_urls(urls)
        
        
        self.vectorizer = TfidfVectorizer(max_features=500, analyzer='char', ngram_range=(3, 5))
        X_text = self.vectorizer.fit_transform(urls)
        text_feature_names = [f'tfidf_{i}' for i in range(X_text.shape[1])]
        X_text_df = pd.DataFrame(X_text.toarray(), columns=text_feature_names)
        
        
        X_combined = pd.concat([X_features, X_text_df], axis=1)
        self.feature_columns = X_combined.columns
        
        # Train model and also print feature importances
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_combined, labels)
        
        # Print top 20 most important features
        print("\nTop 20 most important features:")
        importances = pd.DataFrame({
            'feature': X_combined.columns,
            'importance': self.model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False).head(20)
        for i, row in importances.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        return self

    def predict(self, urls):
        
        features = self.extract_features_from_urls(urls)
        
        
        text_features = self.vectorizer.transform(urls)
        text_feature_names = [f'tfidf_{i}' for i in range(text_features.shape[1])]
        text_features_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names)
        
        
        X = pd.concat([features, text_features_df], axis=1)
        
        
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        X = X[self.feature_columns]
        
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities

    def save(self, filepath):
        """Save the entire model to a single file"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load the entire model from a file"""
        return joblib.load(filepath)


def main():
    
    warnings.filterwarnings('ignore')
    
    
    print("Loading dataset...")
    df = pd.read_csv('urls.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Distribution of labels:\n{df['label'].value_counts()}")
    
    
    y = (df['label'] == 'phishing').astype(int)
    
    
    X_train, X_test, y_train, y_test = train_test_split(df['url'], y, test_size=0.2, random_state=42, stratify=y)
    
    
    print("\nTraining model...")
    detector = PhishingDetector()
    detector.fit(X_train, y_train)
    
    
    print("\nEvaluating model...")
    y_pred, y_prob = detector.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Phishing', 'Phishing']))
    
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Phishing', 'Phishing'], 
                yticklabels=['Not Phishing', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    
    print("\nSaving model...")
    detector.save('phishing_detector.pkl')
    print("Model saved successfully!")
    
    
    print("\nExample predictions:")
    test_urls = [
        "https://google.com",
        "http://amaz0n-secure-login.com",
        "https://facebook.com",
        "http://paypal-account-verify-secure.us",
        "https://www.amazon.com/gp/product/B07X8M3JKS/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1"  # Long legitimate URL
    ]
    
    predictions, probabilities = detector.predict(test_urls)
    for url, pred, prob in zip(test_urls, predictions, probabilities):
        print(f"URL: {url}")
        print(f"Prediction: {'Phishing' if pred == 1 else 'Not Phishing'}")
        print(f"Probability of being phishing: {prob:.4f}")
        print()

if __name__ == "__main__":
    main()