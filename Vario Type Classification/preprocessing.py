import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

class VarioTypes:
    CLASSES = [
        'All New Honda Vario 125 & 150',
        'All New Honda Vario 125 & 150 Keyless',
        'Vario 110',
        'Vario 110 ESP',
        'Vario 160',
        'Vario Techno 110',
        'Vario Techno 125 FI'
    ]
    
    @classmethod
    def get_class_name(cls, index):
        return cls.CLASSES[index]
    
    @classmethod
    def num_classes(cls):
        return len(cls.CLASSES)

class PreprocessImage:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
        # Membuat preprocessing layer yang sama dengan training
        self.preprocessing_layer = tf.keras.Sequential([
            # Standardize size while maintaining aspect ratio
            tf.keras.layers.Resizing(target_size[0], target_size[1]),
            # Center crop to ensure consistent input
            tf.keras.layers.CenterCrop(target_size[0], target_size[1]),
            # Normalize pixel values
            tf.keras.layers.Rescaling(1./255),
        ])

    def load_image(self, image_path):
        """Load and convert image to RGB"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess_image(self, image):
        """
        Preprocess loaded image
        Args:
            image: Loaded image in RGB format
        Returns:
            Preprocessed image
        """
        # Convert to float32
        image = tf.cast(image, tf.float32)
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
            
        # Apply preprocessing
        image = self.preprocessing_layer(image)
        
        return image

    def preprocess(self, image_path):
        """
        Complete preprocessing pipeline from image path
        Args:
            image_path: Path to image file
        Returns:
            Preprocessed image ready for model inference
        """
        # Load image
        image = self.load_image(image_path)
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        return preprocessed
        
    def preprocess_batch(self, image_paths):
        """
        Preprocess multiple images
        Args:
            image_paths: List of image paths
        Returns:
            Batch of preprocessed images
        """
        preprocessed_images = []
        for path in image_paths:
            prep_img = self.preprocess(path)
            preprocessed_images.append(prep_img)
            
        return np.vstack(preprocessed_images)