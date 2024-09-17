import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import shutil
from datetime import datetime
from collections import Counter
from sklearn.metrics import classification_report

# Ensure the current working directory is the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

static_patches_path = os.getcwd() + "/static_patches_input_augmented"
dynamic_patches_path = "../scans/"

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace '4' with the number of cores you want to use

# Initialize ORB
orb = cv2.ORB_create()

# Lists to store features and labels
features = []
labels = []

# Loop through subfolders to load images and labels
for folder, folder_name in [(static_patches_path, 'static'), (dynamic_patches_path, 'dynamic')]:
    print(f"Currently iterating in {folder_name} folder, processing labels: ")
    for label in os.listdir(folder):
        if label != "unsorted" and label != "unidentified":
            print(label, end=", ")
            label_path = os.path.join(static_patches_path, label)
            
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    keypoints, descriptors = orb.detectAndCompute(image, None)
                    if descriptors is not None:
                        features.append(descriptors)
                        labels.append(label)
print()

# Convert the features into a numpy array
features = np.vstack(features)

# Apply K-Means to cluster descriptors across all images
num_clusters = 50
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(features)

# Create histograms for each image based on clusters
histograms = []
for folder, folder_name in [(static_patches_path, 'static'), (dynamic_patches_path, 'dynamic')]:
    print(f"Histograms: currently iterating in {folder_name} folder, processing histograms: ")
    for label in os.listdir(folder):
        if label != "unsorted" and label != "unidentified":
            print(label, end=", ")
            label_path = os.path.join(static_patches_path, label)
            
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    keypoints, descriptors = orb.detectAndCompute(image, None)
                    if descriptors is not None:
                        # Predict clusters for descriptors
                        cluster_indices = kmeans.predict(descriptors)
                        histogram, _ = np.histogram(cluster_indices, bins=np.arange(num_clusters+1))
                        histograms.append(histogram)
print()

# Convert the histograms to a numpy array
features = np.array(histograms)

# Normalize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Check feature and label sizes before splitting
print(f"Number of feature samples: {len(features)}")
print(f"Number of label samples: {len(labels)}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the parameter grid for SVM tuning
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],  # Try different values for C
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],  # Try different values for gamma
    'kernel': ['rbf']  # Use RBF kernel
}

# Initialize the SVM model
svm = SVC(probability=True, class_weight='balanced')

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(svm, param_grid, refit=True, cv=5, verbose=2)

# Train the model using grid search to find the best parameters
grid_search.fit(X_train, y_train)

# Get the best parameters after tuning
print("Best parameters found by GridSearchCV: ", grid_search.best_params_)

# Use the best SVM model to make predictions
best_svm = grid_search.best_estimator_

# Predict on the test data
y_pred = best_svm.predict(X_test)

# Analyze Class Distribution & Model Performance
class_counts = Counter(y_train)
print("Number of samples for each class in your training data: " + str(class_counts))

# Print classification report
print(classification_report(y_test, y_pred))

# Evaluate the accuracy
accuracy = best_svm.score(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the best SVM model after tuning
joblib.dump(best_svm, os.getcwd() + '/new_models/latest/' + 'svm_model.pkl')
joblib.dump(kmeans, os.getcwd() + '/new_models/latest/' + 'kmeans_model.pkl')
joblib.dump(scaler, os.getcwd() + '/new_models/latest/' + 'scaler.pkl')

# Backup the models in the folder corresponding to the current date
copy_source_path = os.path.join(os.getcwd(), "new_models", "latest")
copy_destination_folder = os.path.join(os.getcwd(), "new_models", datetime.today().strftime('%Y-%m-%d'))

# Backup the models into the folder corresponding to the date of generation
shutil.copytree(copy_source_path, copy_destination_folder, dirs_exist_ok=True)
