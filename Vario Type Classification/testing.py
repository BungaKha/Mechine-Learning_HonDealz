import tensorflow as tf
import numpy as np
import os
from preprocessing import PreprocessImage

CLASS_NAMES = [
    'All New Honda Vario 125 & 150',
    'All New Honda Vario 125 & 150 Keyless',
    'Vario 110',
    'Vario 110 ESP',
    'Vario 160',
    'Vario Techno 110',
    'Vario Techno 125 FI'
]

class ModelPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.preprocessor = PreprocessImage()

    def predict(self, image_path):
        print("Making prediction...")
        print("Preprocessing image...")
        input_data = self.preprocessor.preprocess(image_path)
        print("Preprocessing complete.")
        print("1/1 =========== 3s 3s/step")
        predictions = self.model.predict(input_data, verbose=0)
        print(f"Raw prediction values: {predictions.tolist()}")
        return predictions[0]

    def display_predictions(self, predictions):
        top_prediction_idx = np.argmax(predictions)
        top_confidence = predictions[top_prediction_idx] * 100

        print("\n===============================================")
        print("PREDICTIONS:")
        print("===============================================\n")

        print(f"1. {CLASS_NAMES[top_prediction_idx]}")
        print(f"   Confidence: {top_confidence:.2f}%")
        bar_length = int(predictions[top_prediction_idx] * 50)
        progress_bar = '[' + '|' * bar_length + ' ' * (50 - bar_length) + ']'
        print(f"   {progress_bar}\n")

MODEL_PATH = 'PATH/vario-type.keras'
IMAGE_PATH = 'PATH/akdzan.jpg'

try:
    predictor = ModelPredictor(MODEL_PATH)
    predictions = predictor.predict(IMAGE_PATH)
    predictor.display_predictions(predictions)
except Exception as e:
    print(f"Error: {str(e)}")