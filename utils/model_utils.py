import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

class ISLModelPredictor:
    """Utility class for handling sign language model predictions"""
    
    def __init__(self, model_path):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved Keras model
        """
        self.model = load_model(model_path)
        self.img_height = 64
        self.img_width = 64
        
        # Class labels (ensure these match your model's classes)
        self.class_labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
            'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', 
            '5', '6', '7', '8', '9'  # Adjust based on your dataset
        ]
    
    def preprocess_image(self, img_path):
        """
        Preprocess an image for model prediction
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Preprocessed image array ready for model input
        """
        img = image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        return img_array
    
    def preprocess_image_from_array(self, img_array):
        """
        Preprocess an image array for model prediction
        
        Args:
            img_array: NumPy array of image
            
        Returns:
            Preprocessed image array ready for model input
        """
        img = cv2.resize(img_array, (self.img_height, self.img_width))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, img_path):
        """
        Make a prediction from an image file
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Dictionary with predicted class and confidence
        """
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array)
        
        # Get the predicted class index and confidence
        pred_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class_idx])
        
        return {
            'predicted_class': self.class_labels[pred_class_idx],
            'confidence': confidence,
            'all_probabilities': {self.class_labels[i]: float(predictions[0][i]) for i in range(len(self.class_labels))}
        }
    
    def predict_from_array(self, img_array):
        """
        Make a prediction from an image array
        
        Args:
            img_array: NumPy array of image
            
        Returns:
            Dictionary with predicted class and confidence
        """
        processed_img = self.preprocess_image_from_array(img_array)
        predictions = self.model.predict(processed_img)
        
        # Get the predicted class index and confidence
        pred_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class_idx])
        
        return {
            'predicted_class': self.class_labels[pred_class_idx],
            'confidence': confidence,
            'all_probabilities': {self.class_labels[i]: float(predictions[0][i]) for i in range(len(self.class_labels))}
        }
    
    def process_video_frame(self, frame):
        """
        Process a single video frame for sign language detection
        
        Args:
            frame: NumPy array representing a video frame
            
        Returns:
            Dictionary with predicted class and confidence
        """
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess and predict
        return self.predict_from_array(frame_rgb)

# Example usage
if __name__ == "__main__":
    predictor = ISLModelPredictor("../models/isl_rnn_model.keras")
    result = predictor.predict("../test_images/a_sign.jpg")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")