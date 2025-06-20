# 1. Update imports and app initialization
import os
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import glob
import time
import uuid

# 2. Convert the Flask app to a Flask Blueprint
from flask import Blueprint, current_app
app = Flask(__name__)
fapp = Blueprint('fall_detection', __name__, 
                static_folder='static', 
                template_folder='templates',
                url_prefix='/fall-detection')
app.register_blueprint(fapp)
app.secret_key = "fall_detection_secret_key" # Keep this secret in production

# 3. Update the paths to be relative to current directory
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_DIR = 'model/'  # Directory containing all models
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Progress Tracking ---
# Simple dictionary for progress. For concurrent users, a more robust
# solution (like Redis, Celery, or Flask-SocketIO) would be needed.
# Stores progress percentage keyed by a unique task ID.
TASK_PROGRESS = {}
# -------------------------

# Get available models
def get_available_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, MODEL_DIR)
    models = glob.glob(f"{model_path}*.pt")
    return [os.path.basename(model) for model in models]

# Load YOLO model
def load_model(model_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = YOLO(model_path)
    return model

# Check if file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Check if file is an image
def is_image(filename):
    return filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Check if file is a video
def is_video(filename):
    return filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

# Process image with YOLO model
def detect_falls_image(image_path, model, model_name, threshold=0.25):
    # Perform inference with YOLOv8
    results = model(image_path, conf=threshold)

    # Get the original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image file: {image_path}")

    # Get the current directory for proper path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Adjust result path to use the current directory
    result_base, _ = os.path.splitext(os.path.basename(image_path))
    model_identifier = model_name.replace('.pt', '')
    result_filename_base = f"result_{model_identifier}_{result_base}.jpg"
    result_filepath = os.path.join(current_dir, RESULT_FOLDER, result_filename_base)

    detection_data = []

    # Process each detection
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get confidence
            conf = float(box.conf[0])

            # Get class
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Different color based on class (fall vs. no fall)
            color = (0, 0, 255) if class_name.lower() == 'fall' else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Add label with confidence
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Store detection info
            detection_data.append({
                'name': class_name,
                'confidence': conf
            })

    # Save the annotated image (use jpg for web compatibility)
    cv2.imwrite(result_filepath, img)

    return result_filepath, detection_data

# Process video with YOLO model
def detect_falls_video(video_path, model, model_name, task_id=None, threshold=0.25):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # For progress calculation
    
    # Get the current directory for proper path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Adjust result path to use the current directory
    base_filename = os.path.basename(video_path)
    model_identifier = model_name.replace('.pt', '')
    result_filename = os.path.join(current_dir, RESULT_FOLDER, 
                                 f"result_{model_identifier}_{base_filename}")
    
    # Ensure result has .mp4 extension for web compatibility
    if not result_filename.lower().endswith('.mp4'):
        base_name = os.path.splitext(result_filename)[0]
        result_filename = f"{base_name}.mp4"
    
    # Define codec and create VideoWriter object
    # Use H.264 codec for better web browser compatibility
    try:
        # Try avc1 (H.264) first - most web compatible
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  
        out = cv2.VideoWriter(result_filename, fourcc, fps, (width, height))
    except:
        try:
            # Fallback to H264
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(result_filename, fourcc, fps, (width, height))
        except:
            # Last resort - mp4v with explicit .mp4 extension
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_filename, fourcc, fps, (width, height))
    
    # Get detection data for display
    all_detection_data = []
    
    frame_count = 0
    last_progress_update_time = time.time()
    
    # --- Initialize progress ---
    if task_id:
        TASK_PROGRESS[task_id] = 0
    # -------------------------
    
    try:
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Perform inference with YOLOv8
            results = model(frame, verbose=False, conf=threshold)  # Set verbose=False to reduce console spam
            
            frame_detections = []
            
            # Process each detection
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence
                    conf = float(box.conf[0])
                    
                    # Get class
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    # Different color based on class (fall vs. no fall)
                    color = (0, 0, 255) if class_name.lower() == 'fall' else (0, 255, 0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Store detection info
                    frame_detections.append({
                        'name': class_name,
                        'confidence': conf,
                        'frame': frame_count
                    })
            
            # Add frame detections to overall detections
            all_detection_data.extend(frame_detections)
            
            # Write the frame to output video
            out.write(frame)
            
            # --- Update progress periodically ---
            current_time = time.time()
            if task_id and total_frames > 0 and (current_time - last_progress_update_time > 0.5):  # Update every 0.5 seconds
                progress = int((frame_count / total_frames) * 100)
                TASK_PROGRESS[task_id] = progress
                last_progress_update_time = current_time
            # -----------------------------------
    finally:
        # Release everything
        cap.release()
        out.release()
        
        # --- Final Progress Update & Cleanup ---
        if task_id:
            TASK_PROGRESS[task_id] = 100  # Ensure it reaches 100
        # -------------------------------------
    
    # Extract a thumbnail with a unique name
    #thumbnail_path = os.path.join(app.config['RESULT_FOLDER'], 
    #                            f"thumbnail_{model_identifier}_{os.path.basename(video_path).split('.')[0]}.jpg")
    
    # Also update the thumbnail path:
    thumbnail_path = os.path.join(current_dir, RESULT_FOLDER, 
                                f"thumbnail_{model_identifier}_{os.path.basename(video_path).split('.')[0]}.jpg")
    
    cap = cv2.VideoCapture(result_filename)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(thumbnail_path, frame)
    cap.release()
    
    # Summarize detections by class
    summary_data = {}
    for detection in all_detection_data:
        class_name = detection['name']
        if class_name not in summary_data:
            summary_data[class_name] = {
                'count': 0,
                'total_confidence': 0.0,  # Use float for sum
                'max_confidence': 0.0
            }
        
        summary_data[class_name]['count'] += 1
        current_conf = detection['confidence']
        summary_data[class_name]['max_confidence'] = max(summary_data[class_name]['max_confidence'], current_conf)
        summary_data[class_name]['total_confidence'] += current_conf  # Sum before dividing
    
    # Calculate average confidence
    detection_summary = []
    for class_name, data in summary_data.items():
        avg_conf = 0.0
        if data['count'] > 0:
            avg_conf = data['total_confidence'] / data['count']
        
        detection_summary.append({
            'name': class_name,
            'count': data['count'],
            'avg_confidence': f"{avg_conf:.2f}",
            'max_confidence': f"{data['max_confidence']:.2f}"
        })
    
    return result_filename, thumbnail_path, detection_summary
    
@app.route('/')
def index():
    available_models = get_available_models()
    return render_template('index.html', available_models=available_models)


@app.route('/process_detection', methods=['POST'])
def process_detection():
    task_id = None # Initialize task_id
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        selected_model_name = request.form.get('model')
        if not selected_model_name:
            return jsonify({'error': 'Please select a model'})
            
        # Get the threshold value from the form (default to 0.25 if not provided)
        threshold = float(request.form.get('threshold', 0.25))
        # Ensure threshold is within valid range (0.01 to 1.0)
        threshold = max(0.01, min(threshold, 1.0))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(current_dir, UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Load model
            model = load_model(selected_model_name)

            file_type_is_video = is_video(filename)

            # --- Generate Task ID for Videos ---
            if file_type_is_video:
                task_id = str(uuid.uuid4()) # Generate unique ID
                TASK_PROGRESS[task_id] = 0 # Initialize progress
            # -----------------------------------
            
            if is_image(filename):
                result_path, detection_data = detect_falls_image(filepath, model, selected_model_name, threshold)
              
                results = [{'class': item['name'], 'confidence': f"{item['confidence']:.2f}"} for item in detection_data]

                # Construct URLs relative to static folder for the template
                image_file_url = os.path.join('uploads', os.path.basename(filepath)).replace("\\", "/")
                result_file_url = os.path.join('results', os.path.basename(result_path)).replace("\\", "/")

                html_content = render_template('result_partial.html',
                                               image_file=image_file_url,
                                               result_file=result_file_url,
                                               results=results,
                                               is_video=False,
                                               selected_model=selected_model_name)

                return jsonify({'success': True, 'html': html_content, 'is_video': False})

            elif file_type_is_video:
                result_path, thumbnail_path, detection_summary = detect_falls_video(
                    filepath, model, selected_model_name, task_id, threshold
                )
                # Construct URLs relative to static folder for the template
                video_file_url = os.path.join('uploads', os.path.basename(filepath)).replace("\\", "/")
                result_file_url = os.path.join('results', os.path.basename(result_path)).replace("\\", "/")
                thumbnail_url = os.path.join('results', os.path.basename(thumbnail_path)).replace("\\", "/") if thumbnail_path else None

                html_content = render_template('result_partial.html',
                                               video_file=video_file_url,
                                               result_file=result_file_url,
                                               thumbnail=thumbnail_url,
                                               results=detection_summary,
                                               is_video=True,
                                               selected_model=selected_model_name)

                # Return task_id along with results for video
                return jsonify({'success': True, 'html': html_content, 'task_id': task_id, 'is_video': True})

            else: # Should not happen if allowed_file works correctly
                 return jsonify({'error': 'Invalid file type after check'})

        else:
            return jsonify({'error': 'File type not allowed'})

    except FileNotFoundError as e:
         print(f"Error: {e}")
         return jsonify({'error': str(e)})
    except IOError as e:
         print(f"IO Error: {e}")
         return jsonify({'error': f'Error processing video: {str(e)}'})
    except Exception as e:
        print(f"Unexpected Error: {e}") # Log the full error
        import traceback
        traceback.print_exc() # Print stack trace to console
        # --- Ensure progress is cleaned up on error ---
        if task_id and task_id in TASK_PROGRESS:
             del TASK_PROGRESS[task_id]
        # -------------------------------------------
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'})
    finally:
        # --- Clean up progress task if it exists and wasn't handled ---
        # (e.g., if request ends abruptly before video finishes)
        # This is a simple cleanup; more robust systems might handle this differently.
        # if task_id and task_id in TASK_PROGRESS:
        #    del TASK_PROGRESS[task_id] # Can be removed if final cleanup in detect_falls_video is reliable
        pass


# --- New route to get progress ---
@app.route('/get_progress/<task_id>')
def get_progress(task_id):
    progress = TASK_PROGRESS.get(task_id, 0) # Get progress or default to 0 if task_id not found
    # print(f"Polling for {task_id}, Progress: {progress}") # Debugging
    return jsonify({'progress': progress})
# ---------------------------------

# 7. Remove the app.run() call at the bottom, replace with:
# For testing this file directly
if __name__ == '__main__':
    from flask import Flask
    test_app = Flask(__name__)
    test_app.register_blueprint(app)
    test_app.run(debug=True)