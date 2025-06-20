# 1. Update imports and app initialization
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
import os
import cv2
import numpy as np
from PIL import Image
import io
import time
import threading
import uuid
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import json

# 2. Convert the Flask app to a Flask Blueprint
from flask import Blueprint, current_app
app = Flask(__name__)
capp = Blueprint('car_accident', __name__, 
                static_folder='static', 
                template_folder='templates',
                url_prefix='/car_accident')

app.register_blueprint(capp)

# 3. Update the paths to be relative to current directory
# Configuration as variables rather than in app.config
UPLOAD_FOLDER = 'car_accident_detection/static/uploads'
UPLOAD_FOLDER_ACCESS = 'static/uploads'
RESULTS_FOLDER = 'car_accident_detection/static/results'
RESULTS_FOLDER_ACCESS = 'static/results'
MODELS_FOLDER = 'models'  # Folder for storing model files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
FRAME_SKIP = 2  # Process every nth frame for videos
DEFAULT_CONF = 0.25  # Default confidence threshold
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Dictionary to store available models
available_models = {
    'pt': {
        'v1': {
            'path': 'car_accident_detection/models/best.pt',
            'name': 'zero YOLOv8n (PyTorch)',
            'stats': {
                'mAP': 0.775,
                'precision': 0.85,
                'recall': 0.747,
                'f1_score': 0.796,
                'inference_time': '2.4ms per frame',
                'size': '18.7MB',
                'classes': ['Car Accident']
            }
        },
        'v2': {
            'path': 'models/bestr.pt',
            'name': 'zero(realtime) YOLOv8n (PyTorch)',
            'stats': {
                'mAP': 0.796,
                'precision': 0.805,
                'recall': 0.76,
                'f1_score': 0.78,
                'inference_time': '1.5ms per frame',
                'size': '48.5MB',
                'classes': ['Car Accident']
            }
        },
        'v3': {
            'path': 'models/best_old.pt',
            'name': 'zeroOne YOLOv8n (PyTorch)',
            'stats': {
                'mAP': 0.0962,
                'precision': 0.174,
                'recall': 0.107,
                'f1_score': 0.52,
                'inference_time': '19.6ms per frame',
                'size': '48.5MB',
                'classes': ['Car Accident', 'Near Miss']
            }
        }
    },
    'onnx': {
        'v1': {
            'path': 'best.onnx',
            'name': 'zero - YOLOv8n (ONNX)',
            'stats': {
                'mAP': 0.775,
                'precision': 0.85,
                'recall': 0.747,
                'f1_score': 0.796,
                'inference_time': '2.4ms per frame',
                'size': '17.5MB',
                'classes': ['Car Accident']
            }
        },
        'v2': {
            'path': 'best.onnx',
            'name': 'zeroOne - YOLOv8n (ONNX)',
            'stats': {
                'mAP': 0.0962,
                'precision': 0.174,
                'recall': 0.107,
                'f1_score': 0.52,
                'inference_time': '19.6ms per frame',
                'size': '45.2MB',
                'classes': ['Car Accident', 'Near Miss']
            }
        }
    }
}

# Default model
default_model = available_models['pt']['v1']
current_model = None

def load_model(model_type='pt', model_version='v1'):
    """Load and return the specified model"""
    global current_model
    model_info = available_models[model_type][model_version]
    model_path = model_info['path']
    
    # Check if model file exists, if not, use default
    if not os.path.exists(model_path):
        print(f"Warning: Model {model_path} not found. Using default model.")
        model_path = default_model['path']
        model_info = default_model
    
    # Load the model
    model = YOLO(model_path)
    
    # Store current model info
    current_model = {
        'model': model,
        'type': model_type,
        'version': model_version,
        'info': model_info
    }
    
    return model

# Initialize with default model
model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video(filename):
    video_extensions = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def process_image(image_path, result_path, conf_threshold=DEFAULT_CONF, model_obj=None):
    """Process a single image with the model and save results"""
    if model_obj is None:
        model_obj = current_model['model']
        
    # Pass confidence threshold to the model
    results = model_obj(image_path, conf=conf_threshold)
    result_img = results[0].plot()  # Get the annotated image
    cv2.imwrite(result_path, result_img)
    
    # Extract predictions for return
    predictions = []
    for det in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = det
        predictions.append({
            'class': int(cls),
            'confidence': float(conf),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })
    
    return predictions

def process_video(video_path, result_path, task_id, conf_threshold=DEFAULT_CONF, model_obj=None):
    """Process video frame by frame and save the result video"""
    if model_obj is None:
        model_obj = current_model['model']
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        processing_tasks[task_id]['status'] = 'error'
        processing_tasks[task_id]['message'] = 'Failed to open video file'
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer with H.264 codec for better compatibility
    try:
        # Try H.264 codec first (better compatibility)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
        
        # If H.264 fails, try MP4V codec as fallback
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            
        if not out.isOpened():
            # Last resort, try XVID codec which is widely available
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            result_path = result_path.rsplit('.', 1)[0] + '.avi'  # Change extension to .avi
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            processing_tasks[task_id]['result_path'] = result_path  # Update path in processing tasks
    except Exception as e:
        processing_tasks[task_id]['status'] = 'error'
        processing_tasks[task_id]['message'] = f'Failed to initialize video writer: {str(e)}'
        return
    
    frame_count = 0
    processed_count = 0
    accident_frames = 0
    first_accident_frame = None
    first_accident_timestamp = None
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for faster processing, but process at least 1 frame per second
        if frame_count % FRAME_SKIP == 0 or frame_count == 1:
            # Process frame with model, passing the confidence threshold
            results = model_obj(frame, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            processed_count += 1
            
            # Check for detections
            if len(results[0].boxes) > 0:
                accident_frames += 1
                
                # Store first frame with accident
                if first_accident_frame is None:
                    first_accident_frame = frame_count
                    # Calculate timestamp
                    seconds = frame_count / fps
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    first_accident_timestamp = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        else:
            # Write original frame for skipped frames
            out.write(frame)
        
        # Update progress
        progress = int((frame_count / total_frames) * 100)
        processing_tasks[task_id]['progress'] = progress
    
    # Release resources
    cap.release()
    out.release()
    
    # Update task status
    processing_tasks[task_id]['status'] = 'completed'
    processing_tasks[task_id]['processed_frames'] = processed_count
    processing_tasks[task_id]['total_frames'] = frame_count
    processing_tasks[task_id]['accident_frames'] = accident_frames
    processing_tasks[task_id]['first_accident_frame'] = first_accident_frame
    processing_tasks[task_id]['accident_timestamp'] = first_accident_timestamp

# Status tracking for processing tasks
processing_tasks = {}

@app.route('/')
def index():
    return render_template('index.html', models=available_models)

@app.route('/model_stats/<model_type>/<model_version>')
def model_stats(model_type, model_version):
    if model_type in available_models and model_version in available_models[model_type]:
        model_info = available_models[model_type][model_version]
        return render_template('model_stats.html', model_info=model_info, model_type=model_type, model_version=model_version)
    else:
        return "Model not found", 404

@app.route('/set_model', methods=['POST'])
def set_model():
    try:
        data = request.json
        model_type = data.get('model_type', 'pt')
        model_version = data.get('model_version', 'v1')
        
        # Validate input
        if model_type not in available_models or model_version not in available_models[model_type]:
            return jsonify({'error': 'Invalid model selection'}), 400
        
        # Load the selected model
        load_model(model_type, model_version)
        
        # Return success with model info
        return jsonify({
            'success': True,
            'model': {
                'type': model_type,
                'version': model_version,
                'name': available_models[model_type][model_version]['name'],
                'stats': available_models[model_type][model_version]['stats']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get confidence threshold from form data or use default
    try:
        conf_threshold = float(request.form.get('confidence', DEFAULT_CONF))
        # Clamp conf_threshold between 0 and 1
        conf_threshold = max(0.0, min(1.0, conf_threshold))
    except (ValueError, TypeError):
        conf_threshold = DEFAULT_CONF
    
    # Get model selection from form data
    model_type = request.form.get('model_type', 'pt')
    model_version = request.form.get('model_version', 'v1')
    
    # Validate and load model if different from current
    if (current_model['type'] != model_type or current_model['version'] != model_version):
        if model_type in available_models and model_version in available_models[model_type]:
            load_model(model_type, model_version)
        else:
            return jsonify({'error': 'Invalid model selection'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        print("UPLOAD PATH --> ",upload_path)
        file.save(upload_path)
        
        # Add current model info to response data
        model_info = {
            'name': current_model['info']['name'],
            'type': current_model['type'],
            'version': current_model['version'],
            'stats': current_model['info']['stats']
        }
        
        if is_video(unique_filename):
            # For videos, start processing in background
            result_path = os.path.join(RESULTS_FOLDER, f"{unique_filename}")
            task_id = str(uuid.uuid4())
            
            processing_tasks[task_id] = {
                'status': 'processing',
                'progress': 0,
                'file_type': 'video',
                'original_path': upload_path,
                'result_path': result_path,
                'confidence': conf_threshold,
                'model_info': model_info
            }
            
            # Start processing in a separate thread
            thread = threading.Thread(
                target=process_video, 
                args=(upload_path, result_path, task_id, conf_threshold, current_model['model'])
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'task_id': task_id,
                'file_type': 'video',
                'message': 'Video processing started',
                'confidence': conf_threshold,
                'model_info': model_info
            })
        else:
            # For images, process immediately
            result_path = os.path.join(RESULTS_FOLDER, unique_filename)
            predictions = process_image(upload_path, result_path, conf_threshold, current_model['model'])
            
            return jsonify({
                'file_type': 'image',
                'original_path': f"/car_accident/static/uploads/{unique_filename}",
                'result_path': f"/car_accident/static/results/{unique_filename}",
                'predictions': predictions,
                'confidence': conf_threshold,
                'model_info': model_info
            })
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/check_status/<task_id>')
def check_status(task_id):
    if task_id in processing_tasks:
        return jsonify(processing_tasks[task_id])
    return jsonify({'error': 'Task not found'}), 404

@app.route('/results/<task_id>')
def get_results(task_id):
    if task_id in processing_tasks and processing_tasks[task_id]['status'] == 'completed':
        task = processing_tasks[task_id]
        
        # Get proper relative paths for URLs
        original_file = os.path.basename(task['original_path'])
        result_file = os.path.basename(task['result_path'])
        
        # Determine if accident occurred based on parameters
        accident_detected = False
        accident_severity = "None"
        
        # Logic to determine if accident occurred
        processed_frames = task.get('processed_frames', 0)
        accident_frames = task.get('accident_frames', 0)
        confidence = task.get('confidence', DEFAULT_CONF)
        
        # Accident detection criteria
        accident_frame_ratio = accident_frames / processed_frames if processed_frames > 0 else 0
        
        if accident_frames > 0 and confidence > 0.75 and accident_frame_ratio > 0.05:
            accident_detected = True
            
            # Determine severity based on confidence and accident frame ratio
            if confidence > 0.9 and accident_frame_ratio > 0.2:
                accident_severity = "Severe"
            elif confidence > 0.8 or accident_frame_ratio > 0.1:
                accident_severity = "Moderate"
            else:
                accident_severity = "Minor"
        
        return jsonify({
            'status': 'completed',
            'file_type': task['file_type'],
            'original_path': f"/car_accident/static/uploads/{original_file}",
            'result_path': f"/car_accident/static/results/{result_file}",
            'processed_frames': processed_frames,
            'total_frames': task.get('total_frames', 0),
            'accident_frames': accident_frames,
            'confidence': confidence,
            'accident_detected': accident_detected,
            'accident_severity': accident_severity,
            'model_info': task.get('model_info', {}),
            'accident_details': {
                'frame_ratio': round(accident_frame_ratio * 100, 2),
                'first_detection_frame': task.get('first_accident_frame', 0),
                'timestamp': task.get('accident_timestamp', '00:00:00')
            }
        })
    
    return jsonify({'error': 'Results not ready or task not found'}), 404

# Routes to serve static files
@app.route('/static/uploads/<path:filename>')
def serve_upload(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(current_dir, UPLOAD_FOLDER_ACCESS)
    return send_from_directory(uploads_dir, filename)

@app.route('/static/results/<path:filename>')
def serve_result(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, RESULTS_FOLDER_ACCESS)
    return send_from_directory(results_dir, filename)

# 6. Remove the app.run() call at the bottom, replace with:
# For testing this file directly
if __name__ == '__main__':
    from flask import Flask
    test_app = Flask(__name__)
    test_app.register_blueprint(app)
    test_app.run(debug=True)