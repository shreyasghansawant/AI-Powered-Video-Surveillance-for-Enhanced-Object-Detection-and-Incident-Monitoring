from flask import Flask, render_template, redirect, url_for
import os
from car_accident_detection.app import app as car_accident_app
from fall_detection.app import app as fall_detection_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.exceptions import NotFound

app = Flask(__name__)

# Configuration for the main app
app.secret_key = "combined_apps_secret_key"

# Set subdirectories for each application
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app,
    {
        '/car_accident': car_accident_app,
        '/fall_detection': fall_detection_app
    }
)

# Ensure all necessary directories exist
@app.before_request
def create_directories():
    # Main app directories
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Car accident detection directories
    os.makedirs('car_accident_detection/static/uploads', exist_ok=True)
    os.makedirs('car_accident_detection/static/results', exist_ok=True)
    os.makedirs('car_accident_detection/models', exist_ok=True)
    
    # Fall detection directories
    os.makedirs('fall_detection/static/uploads', exist_ok=True)
    os.makedirs('fall_detection/static/results', exist_ok=True)
    os.makedirs('fall_detection/model', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/car_accident_redirect')
def car_accident_redirect():
    return redirect('/car_accident/')

@app.route('/fall_detection_redirect')
def fall_detection_redirect():
    return redirect('/fall_detection/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)