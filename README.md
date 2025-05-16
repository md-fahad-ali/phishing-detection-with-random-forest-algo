# Phishing URL Detection Model

This project implements a machine learning model to detect phishing URLs using various features extracted from the URL string.

## Features

The model extracts the following types of features from URLs:

1. **Structural features**:
   - URL length, domain length, path length
   - Number of subdomains, path depth
   - Special character counts (dots, hyphens, etc.)

2. **Content-based features**:
   - Presence of suspicious words (login, verify, account, etc.)
   - Domain has digits
   - Presence of IP address in URL

3. **Security indicators**:
   - HTTPS protocol usage
   - Non-standard ports
   - Unusual TLDs (xyz, top, club, etc.)

4. **Statistical features**:
   - Shannon entropy of URL (measures randomness)
   - TF-IDF character n-grams

## Model

The implementation uses a Random Forest classifier which is well-suited for this type of classification task due to:
- Handling a mix of feature types
- Resistance to overfitting
- Ability to rank feature importance

## Quick Start

The easiest way to run the project is using the provided shell script:

```bash
./run.sh
```

This script will:
1. Create a virtual environment if needed
2. Install all required dependencies
3. Train the model if needed
4. Start the API server

## Manual Installation

If you prefer to set up manually:

### Installation

```bash
# Create a virtual environment
python3 -m venv fresh_venv

# Activate the environment
source fresh_venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Running the Model

```bash
python phishing_detection.py
```

The script will:
1. Load the dataset from `urls.csv`
2. Extract features from the URLs
3. Train a Random Forest model
4. Evaluate on a test set
5. Display accuracy metrics and feature importance
6. Run predictions on example URLs

### Running the API

```bash
python phishing_api.py
```

This will start a Flask API server at http://localhost:5000 where you can test URL detection.

### Testing the API

```bash
python test_api.py
```

This will run a simple test script to verify the API is working correctly.

## Browser Extension

A Chrome extension is included in the `browser_extension` directory. To use it:

1. Go to chrome://extensions/
2. Enable "Developer mode"
3. Click "Load unpacked" and select the browser_extension folder
4. Make sure the API server is running for full functionality

## Performance

The model typically achieves 95%+ accuracy on the test set, with good precision and recall for both phishing and legitimate URL classes.

## Dataset

The dataset used is a collection of labeled URLs:
- Phishing URLs collected from various sources
- Legitimate URLs from popular websites

The dataset is contained in `urls.csv` with two columns: `url` and `label` (phishing or not-phishing).

## Future Improvements

- Implement real-time URL checking capability
- Add more advanced features (WHOIS data, webpage content)
- Experiment with deep learning approaches
- Create a web interface for easy usage

## Troubleshooting

If you encounter issues with dependencies, make sure you're using a virtual environment and have the latest pip, setuptools, and wheel packages installed:

```bash
python3 -m venv fresh_venv
source fresh_venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Verifying API Status

You can verify if the API server is running correctly using the provided test script:

```bash
python test_api_running.py
```

This script will:
1. Check if the API is running and accessible
2. Start the API server if it's not running (with your permission)
3. Test example URLs to verify the model is working correctly

If you encounter any "module not found" errors, make sure all dependencies are installed:

```bash
pip install -r requirements.txt
``` # phishing-detection-with-random-forest-algo
