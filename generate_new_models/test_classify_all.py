import cv2
import numpy as np
import joblib
import os

# Ensure the current working directory is the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load models
svm = joblib.load('new_models/latest/svm_model.pkl')
kmeans = joblib.load('new_models/latest/kmeans_model.pkl')
scaler = joblib.load('new_models/latest/scaler.pkl')

# Initialize ORB
orb = cv2.ORB_create()

def extract_features(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    if descriptors is None:
        return None
    
    # Predict the clusters for the descriptors
    cluster_indices = kmeans.predict(descriptors)
    
    # Create histogram
    histogram, _ = np.histogram(cluster_indices, bins=np.arange(len(kmeans.cluster_centers_) + 1))
    
    return histogram

def classify_image(image_path):
    # Extract features
    histogram = extract_features(image_path)
    
    if histogram is None:
        print(f"Cannot extract features from image {image_path}")
        return None
    
    # Normalize the histogram
    histogram = histogram.reshape(1, -1)  # Reshape for scaler
    normalized_histogram = scaler.transform(histogram)
    
    # Predict the class
    prediction = svm.predict(normalized_histogram)
    prediction_proba = svm.predict_proba(normalized_histogram)
    
    return prediction[0], prediction_proba[0]

def main(image_directory):
    for image_file in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_file)
        predicted_class, proba = classify_image(image_path)
        prediction_proba = dict(zip(svm.classes_, proba))
        
        if predicted_class is not None:
            print(f"Image: {image_file}")
            print(f"Predicted Class: {predicted_class}")
            
            # Sort the probabilities and keep only the top 4
            top_4_predictions = sorted(prediction_proba.items(), key=lambda x: x[1], reverse=True)[:4]
            
            # Print the top 4 class probabilities in percentage
            print("Top 4 Class Probabilities:")
            for class_name, prob in top_4_predictions:
                print(f"{class_name}: {prob * 100:.2f}%", end=', ')
            print("\n")

if __name__ == "__main__":
    # Path to the directory containing images to classify
    image_directory = 'test_patches_to_classify'
    main(image_directory)