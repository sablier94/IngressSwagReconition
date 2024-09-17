import cv2
from PIL import Image
from io import BytesIO
import joblib
import base64
import numpy as np
import os

# Ensure the current working directory is the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Paths to model files
MODEL_PATHS = {
    'svm_model': os.getcwd() + "\\generate_new_models\\new_models\\latest\\svm_model.pkl",
    'kmeans_model': os.getcwd() + '\\generate_new_models\\new_models\\latest\\kmeans_model.pkl',
    'scaler': os.getcwd() +'\\generate_new_models\\new_models\\latest\\scaler.pkl'
}

# Initialize ORB detector
orb = cv2.ORB_create()

def load_models():
    """Load models from paths specified."""
    svm = joblib.load(MODEL_PATHS['svm_model'])
    kmeans = joblib.load(MODEL_PATHS['kmeans_model'])
    scaler = joblib.load(MODEL_PATHS['scaler'])
    return svm, kmeans, scaler

def preprocess_image(image_bytes):
    """Preprocess image: Convert to grayscale without resizing."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error opening image: {str(e)}")

    # Convert PIL image to NumPy array
    image_np = np.array(image)
    # Convert RGB to BGR (OpenCV uses BGR by default)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Convert to grayscale
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_gray

def extract_features(image_np, kmeans, num_clusters=50):
    """Extract features from the image using ORB and KMeans."""
    keypoints, descriptors = orb.detectAndCompute(image_np, None)
    if descriptors is None:
        return np.zeros(num_clusters)  # Return zero vector if no descriptors found
    cluster_indices = kmeans.predict(descriptors)
    histogram, _ = np.histogram(cluster_indices, bins=np.arange(num_clusters + 1))
    return histogram

def lambda_handler(event, context):
    """Lambda function handler."""
    try:
        # Load models
        svm, kmeans, scaler = load_models()
        
        # Read image from request body
        image_data = event.get('body')
        if not image_data:
            raise ValueError("No image data found in request body")
        
        # Convert base64 encoded image data to bytes
        image_bytes = base64.b64decode(image_data)
        
        # Preprocess image
        test_image = preprocess_image(image_bytes)
        
        # Extract features
        test_features = extract_features(test_image, kmeans)
        
        # Normalize features
        test_features = scaler.transform([test_features])
        
        # Predict class and probabilities
        prediction = svm.predict(test_features)
        prediction_proba = svm.predict_proba(test_features)[0]  # Get probabilities for the first sample
        
        # Handle class labels that might be strings
        predicted_class_label = prediction[0]
        if isinstance(predicted_class_label, str):
            # Convert class labels to indices
            class_labels = svm.classes_
            class_index = np.where(class_labels == predicted_class_label)[0][0]
            predicted_proba = prediction_proba[class_index]
        else:
            predicted_proba = prediction_proba[int(predicted_class_label)]
        
        # Format probabilities as percentages
        prob_percentage = predicted_proba * 100
        
        # Return prediction with probabilities and the image id (passed through the handler context)
        return {
            'statusCode': 200,
            'body': {
                'class': predicted_class_label,
                'probability': int(prob_percentage),
                'scanid': context  # Include the image unique_id (filename) in the response
            }
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e)
        }
