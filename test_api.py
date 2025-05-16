import requests
import json

def test_api():
    # Define API endpoints
    api_url = "http://localhost:5000/api/check"
    batch_api_url = "http://localhost:5000/api/check-batch"
    
    # Test single URL check
    print("Testing single URL check...")
    test_urls = [
        "https://google.com",  # Not phishing
        "http://pub-f7076e6b96664daf90ae8357ef8b9110.r2.dev/DF1.html",  # Likely phishing
    ]
    
    for url in test_urls:
        response = requests.post(api_url, json={"url": url})
        if response.status_code == 200:
            result = response.json()
            print(f"URL: {result['url']}")
            print(f"Is Phishing: {result['is_phishing']}")
            print(f"Probability: {result['phishing_probability']:.4f}")
            print(f"Prediction: {result['prediction']}")
            print()
        else:
            print(f"Error checking {url}: {response.status_code} - {response.text}")
    
    # Test batch URL check
    print("\nTesting batch URL check...")
    batch_response = requests.post(batch_api_url, json={"urls": test_urls})
    
    if batch_response.status_code == 200:
        results = batch_response.json()['results']
        for result in results:
            print(f"URL: {result['url']}")
            print(f"Is Phishing: {result['is_phishing']}")
            print(f"Probability: {result['phishing_probability']:.4f}")
            print(f"Prediction: {result['prediction']}")
            print()
    else:
        print(f"Error with batch check: {batch_response.status_code} - {batch_response.text}")
        
if __name__ == "__main__":
    test_api() 