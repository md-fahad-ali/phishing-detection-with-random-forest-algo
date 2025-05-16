#!/usr/bin/env python
"""
Script to check if the Phishing Detection API is running.
This is useful for troubleshooting when the extension can't connect to the API.
"""
import requests
import sys
import time
import os
import signal
import subprocess
from urllib.parse import urlparse

API_URL = "http://localhost:5000/api/check"
API_SERVER_SCRIPT = "phishing_api.py"
VIRTUAL_ENV = "fresh_venv"

def check_api_running():
    """Check if the API is running and accessible"""
    try:
        response = requests.post(
            API_URL, 
            json={"url": "https://example.com"}, 
            timeout=2
        )
        if response.status_code == 200:
            print(f"✅ API is running at {API_URL}")
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"❌ API returned error status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to API at {API_URL}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Connection to API timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking API: {e}")
        return False

def start_api_server():
    """Start the API server if it's not running"""
    print("🚀 Starting API server...")
    
    # Get absolute path to the API script
    script_path = os.path.join(os.getcwd(), API_SERVER_SCRIPT)
    venv_path = os.path.join(os.getcwd(), VIRTUAL_ENV, "bin", "python")
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Cannot find {API_SERVER_SCRIPT} in the current directory")
        return None
    
    if not os.path.exists(venv_path):
        print(f"❌ Error: Virtual environment not found at {VIRTUAL_ENV}")
        return None
        
    try:
        # Start the API server as a separate process
        process = subprocess.Popen(
            [venv_path, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setpgrp  # This makes the process group leader
        )
        
        # Wait a few seconds for the server to start
        time.sleep(3)
        
        print(f"✅ API server started with PID {process.pid}")
        print("📝 Server output:")
        
        # Get initial output (non-blocking)
        output, error = process.communicate(timeout=0.1)
        if output:
            print(output.strip())
        if error:
            print(f"⚠️ Error output: {error.strip()}")
        
        return process
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def test_url(url):
    """Test a specific URL with the API"""
    try:
        # Validate URL
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            print(f"❌ Invalid URL: {url}")
            return
            
        print(f"🔍 Testing URL: {url}")
        response = requests.post(API_URL, json={"url": url})
        
        if response.status_code == 200:
            data = response.json()
            is_phishing = data.get("is_phishing", False)
            probability = data.get("phishing_probability", 0)
            
            if is_phishing:
                print(f"⚠️ WARNING: {url} is likely a PHISHING site!")
                print(f"Probability: {probability:.2%}")
            else:
                print(f"✅ {url} appears to be legitimate")
                print(f"Probability of being phishing: {probability:.2%}")
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error testing URL: {e}")

if __name__ == "__main__":
    api_running = check_api_running()
    process = None
    
    if not api_running:
        print("\nThe API server isn't running. Would you like to start it? (y/n)")
        choice = input("> ").strip().lower()
        if choice in ('y', 'yes'):
            process = start_api_server()
            if process:
                # Check again
                time.sleep(2)
                api_running = check_api_running()
    
    if api_running:
        # If a URL is provided, test it
        if len(sys.argv) > 1:
            test_url(sys.argv[1])
        else:
            # Test some example URLs
            print("\n🔍 Testing example URLs:")
            test_urls = [
                "https://google.com",
                "http://amaz0n-secure-login.com", 
                "https://facebook.com",
                "http://pub-f7076e6b96664daf90ae8357ef8b9110.r2.dev/DF1.html"  # A known phishing URL from the dataset
            ]
            for url in test_urls:
                test_url(url)
                time.sleep(1)  # Small delay between requests
                
        print("\n✅ API testing complete")
    else:
        print("\n❌ Could not connect to the API server.")
        print("Please make sure to run the API server using:")
        print("source fresh_venv/bin/activate && python phishing_api.py")
    
    # If we started the server, ask if we should stop it
    if process and process.poll() is None:
        print("\nWould you like to stop the API server? (y/n)")
        choice = input("> ").strip().lower()
        if choice in ('y', 'yes'):
            try:
                # Kill the process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                print("✅ API server stopped")
            except Exception as e:
                print(f"❌ Error stopping API server: {e}")
                print("You may need to manually stop it using: kill {process.pid}") 