import requests
import json

def test_api():
    # Define API endpoints
    api_url = "http://localhost:5000/api/check"
    batch_api_url = "http://localhost:5000/api/check-batch"
    
    # Test single URL check
    print("Testing single URL check...")
    test_urls = [
        "https://www.amazon.co.uk/ap/signin?clientContext=260-5863580-6761515&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.amazon.co.uk%2Fgp%2Fvideo%2Fauth%2Freturn%2Fref%3Dav_auth_ap%3F_t%3D1sg2NyuqEtZZ_rvx3xtCnoqDskXFQwcj9Df6lP8l0B1KRHAAAAAQAAAABoJrMmcmF3AAAAAPgWC9WfHH8iB-olH_E9xQ%26location%3D%2Fgp%2Fvideo%2Fontv%2Fcode%3Fref_%253Datv_auth_red_aft&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&accountStatusPolicy=P1&openid.assoc_handle=gbflex&openid.mode=checkid_setup&countryCode=GB&language=en_GB&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0",  # Not phishing
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