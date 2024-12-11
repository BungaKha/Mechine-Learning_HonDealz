# HonDealz Price Prediction and Classification

## About The Project
HonDealz is a machine learning project focused on Honda Vario motorcycle price prediction and type classification. The project combines web scraping, data processing, and machine learning to provide accurate price estimates and motorcycle type classification.

## Project Structure
```
├── Collecting Dataset/
│   ├── collectingData.ipynb        # Data collection notebook
│   ├── dataset_capstone sebelum.csv
│   ├── dataset_momotor.csv
│   ├── dataset_olx.csv
│   └── chromedriver.exe
│
├── Data Processing/
│   ├── Processing data.ipynb       # Data processing notebook
│   ├── cleaningdata.ipynb         # Data cleaning procedures
│   ├── dataset_capstone.csv       # Final processed dataset
│   └── chromedriver.exe
│
├── Scraping Gambar/
│   ├── Scraping Gambar.ipynb      # Image scraping notebook
│   └── scraping_vario_type.ipynb  # Vario type scraping
│
├── Vario Price Prediction Model/
│   ├── optimized_price_model_with.joblib  # Trained price prediction model
│   ├── preprocessing.py           # Data preprocessing module
│   ├── tensorflowtraining.ipynb  # Model training notebook
│   └── vario_price_predictor.tflite      # TFLite model
│
└── Vario Type Classification/
    ├── crop-image.py             # Image preprocessing
    ├── example.jpg               # Sample image
    ├── preprocessing.py          # Classification preprocessing
    ├── testing.py               # Model testing script
    └── vario_classifier.tflite   # Classification model
```

## Features
1. **Price Prediction**
   - Machine learning model to predict Honda Vario prices
   - Multiple model ensemble (Random Forest, XGBoost, Gradient Boosting)
   - TensorFlow implementation for deep learning approach

2. **Type Classification**
   - Image-based Vario type classification
   - TFLite model for mobile deployment
   - Support for multiple Vario variants

3. **Data Collection**
   - Web scraping from multiple sources
   - Automated data gathering
   - Image collection capabilities

## Prerequisites
- Python 3.8+
- TensorFlow 2.15.0
- Scikit-learn
- Pandas
- NumPy
- Selenium (for web scraping)
- Chrome WebDriver

## Usage
### Price Prediction
```python
from price_prediction.preprocessing import MotorcyclePricePreprocessor

# Initialize preprocessor
preprocessor = MotorcyclePricePreprocessor()

# Prepare data
data = {
    'model': 'Vario 125',
    'year': 2020,
    'mileage': '15000 km',
    'location': 'Jakarta Selatan',
    'tax': 'hidup',
    'seller_type': 'private'
}

# Get prediction
result = preprocessor.predict(data)
```

### Type Classification
```python
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="vario_classifier.tflite")
interpreter.allocate_tensors()

# Process image and get prediction
# (See testing.py for complete implementation)
```

## Contact
Project Link: [https://github.com/BungaKha/Menchine-Learning_HonDealz](https://github.com/BungaKha/Menchine-Learning_HonDealz)

## Acknowledgments
* [TensorFlow](https://www.tensorflow.org/)
* [Scikit-learn](https://scikit-learn.org/)
* [Selenium](https://www.selenium.dev/)
