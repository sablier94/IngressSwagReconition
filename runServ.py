from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import base64
import uuid
from datetime import datetime
import shutil
import subprocess
import json

# Import your Lambda function module
from SwagAnalysis import lambda_handler

# Ensure the current working directory is the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add the local `python` directory to the system path
sys.path.append(os.path.join(os.getcwd(), "python"))

app = Flask(__name__)

# Enable CORS for all domains
CORS(app)

# Ensure the scans/unsorted directory exists
unsorted_folder = os.path.join(os.getcwd(), 'scans', 'unsorted')
unidentified_folder = os.path.join(os.getcwd(), 'scans', 'unidentified')
os.makedirs(unsorted_folder, exist_ok=True)
os.makedirs(unidentified_folder, exist_ok=True)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/admin')
def serve_admin():
    return send_from_directory('static', 'admin.html')

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        # 1. Get creation date of kmeans_model.pkl
        kmeans_model_path = 'generate_new_models/new_models/latest/kmeans_model.pkl'
        if os.path.exists(kmeans_model_path):
            creation_time = os.path.getmtime(kmeans_model_path)
            creation_time = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
        else:
            creation_time = 'File not found'

        # 2. Count the number of files in scans/unidentified
        unidentified_path = 'scans/unidentified'
        if os.path.exists(unidentified_path):
            num_unidentified = len([f for f in os.listdir(unidentified_path) if os.path.isfile(os.path.join(unidentified_path, f))])
        else:
            num_unidentified = 0

        # 3. Count the number of files in scans/unsorted
        unsorted_path = 'scans/unsorted'
        if os.path.exists(unsorted_path):
            num_unsorted = len([f for f in os.listdir(unsorted_path) if os.path.isfile(os.path.join(unsorted_path, f))])
        else:
            num_unsorted = 0

        # 4. Count the number of folders in scans (excluding unsorted and unidentified)
        scans_path = 'scans'
        if os.path.exists(scans_path):
            num_folders = len([d for d in os.listdir(scans_path)
                               if os.path.isdir(os.path.join(scans_path, d)) and d not in ['unsorted', 'unidentified']])
        else:
            num_folders = 0

        # Return stats as JSON
        return jsonify({
            'kmeans_model_creation_date': creation_time,
            'num_unidentified_files': num_unidentified,
            'num_unsorted_files': num_unsorted,
            'num_folders_in_scans': num_folders
        }), 200

    except Exception as e:
        # In case of error, return the error message
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/getclasses', methods=['GET'])
def get_classes():
    try:
        scans_path = 'scans'
        if os.path.exists(scans_path):
            folders = [d for d in os.listdir(scans_path)
                               if os.path.isdir(os.path.join(scans_path, d)) and d not in ['unsorted', 'unidentified']]
        else:
            folders = "Fetching error"

        # Return stats as JSON
        return jsonify({
            'classes': folders
        }), 200

    except Exception as e:
        # In case of error, return the error message
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Route for retraining
@app.route('/retrain', methods=['GET'])
def serve_retrain():
    try:
        # Run the external script and capture the output
        result = subprocess.run(
            ['python3', 'generate_new_models/model_generation_for_patches_v3_dynamic.py'],
            capture_output=True, text=True
        )

        # Check if the script ran successfully
        if result.returncode == 0:
            output = result.stdout  # Get the stdout from the script
        else:
            output = result.stderr  # Capture any error output

        # Return the output as JSON
        return jsonify({
            'status': 'success' if result.returncode == 0 else 'error',
            'message': output
        }), 200 if result.returncode == 0 else 500

    except Exception as e:
        # If there's any error during execution, return it as JSON
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400

    image_data = request.json['image']
    
    # Decode the base64 image data
    image_data = base64.b64decode(image_data)
    
    # Generate a unique filename using timestamp and UUID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())
    filename = f'{timestamp}_{unique_id}.jpg'
    
    # Define the path to save the image in the scans/unsorted folder
    image_path = os.path.join(unsorted_folder, filename)
    
    # Save the image
    with open(image_path, 'wb') as image_file:
        image_file.write(image_data)
    
    # Create the event for the lambda handler
    event = {
        'body': base64.b64encode(image_data).decode('utf-8')
    }
    
    # Call the lambda handler with the event
    response = lambda_handler(event, unique_id)
    print(response)
    try:
        # Check if response['body'] is already a dict, if not, load as JSON
        if isinstance(response['body'], str):
            response_body = json.loads(response['body'])
        else:
            response_body = response['body']
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Response body: {response['body']}")

    # Now, assign 'scanid' to the response body
    response_body['scanid'] = filename.rsplit('.', 1)[0]

    # Ensure the response body is updated with the new 'scanid' and converted back to JSON string
    response['body'] = json.dumps(response_body)

    # Return the response as JSON (this ensures proper format for the frontend)
    return jsonify(response_body)

@app.route('/approve', methods=['POST'])
def approve():
    if not request.json or 'scanid' not in request.json or 'class' not in request.json:
        return jsonify({"error": "scanid and class are required"}), 400

    scanid = request.json['scanid']
    classification = request.json['class']
    
    # Define the paths for the source file in unsorted and the destination class folder
    source_path = os.path.join(unsorted_folder, f"{scanid}.jpg")
    class_folder = os.path.join(os.getcwd(), 'scans', classification)
    os.makedirs(class_folder, exist_ok=True)  # Create the class folder if it doesn't exist
    destination_path = os.path.join(class_folder, f"{scanid}.jpg")
    
    # Move the image to the class folder
    if os.path.exists(source_path):
        shutil .move(source_path, destination_path)
        return jsonify({"message": f"Image {scanid}.jpg moved to {classification} folder."}), 200
    else:
        return jsonify({"error": f"Image {scanid}.jpg not found."}), 404

@app.route('/reject', methods=['POST'])
def reject():
    if not request.json or 'scanid' not in request.json:
        return jsonify({"error": "scanid is required"}), 400

    scanid = request.json['scanid']
    
    # Define the paths for the source file in unsorted and the destination unidentified folder
    source_path = os.path.join(unsorted_folder, f"{scanid}.jpg")
    destination_path = os.path.join(unidentified_folder, f"{scanid}.jpg")
    
    # Move the image to the unidentified folder
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        return jsonify({"message": f"Image {scanid}.jpg moved to unidentified folder."}), 200
    else:
        return jsonify({"error": f"Image {scanid}.jpg not found."}), 404

if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        #port=5000,
        port=443, 
        ssl_context=('selfsigned.crt', 'selfsigned.key')
    )
