# AI-Powered Video Surveillance for Enhanced Object Detection and Incident Monitoring

> 🎓 Final Year Project | ISBM College of Engineering, Department of AI & Data Science  
> 📅 Academic Year: 2024–2025  
> 👨‍💻 Authors: Shreyas Ghansawant, Atharva Thokal, Tanishq Ladde  
> 🧑‍🏫 Guide: Prof. Nikita Khawase

---

## 🔍 Overview

This project is an AI-enabled video surveillance system that uses state-of-the-art object detection models (YOLOv8) to detect **car crashes** and **human falls** from uploaded videos or images. The goal is to automate incident detection for better safety and quicker emergency response.

The system integrates deep learning with a web application built using **Python Flask**, offering an easy-to-use platform where users can upload footage and receive real-time annotated incident reports.

---

## 🧠 Methodology

### 1. **Data Collection and Annotation**
- Datasets curated from **Roboflow**, public sources (Open Images, AI City Challenge)
- Manually labeled incidents (`car crash`, `fall`, `standing person`)
- Separate datasets for each type of incident

### 2. **Data Preprocessing and Augmentation**
- Resizing (640x640), normalization
- Augmentations: rotation, flipping, brightness, contrast, occlusion

### 3. **Model Training (YOLOv8)**
- Trained 3 versions for each task (crash and fall)
- Best version (v3) selected based on evaluation metrics
- Training Environment: Google Colab (with GPU)
- Hyperparameters:  
  - Epochs: 100  
  - Batch size: 16  
  - Learning rate: 0.001  

### 4. **Model Integration**
- Flask handles file upload and inference
- OpenCV for frame-by-frame annotation
- Annotated video/image returned to user

### 5. **Web Interface**
- Upload video/image
- Get real-time annotated output
- Download option available

---

## 🎯 Features

- 🔍 YOLOv8 object detection
- 🎥 Upload videos/images
- 🧠 Real-time incident classification (Crash & Fall)
- 🌐 Flask-based web application
- 📥 Annotated output with bounding boxes and class labels
- 🛠️ Easily expandable and modular

---

## 🧪 Results

### 📊 Crash Detection (Best Model: V1)
| Metric     | Value |
|------------|-------|
| mAP@0.5    | 0.77  |
| Precision  | 0.85  |
| Recall     | 0.74  |
| F1 Score   | 0.79  |

### 📊 Fall Detection (Best Model: V3)
| Metric     | Value |
|------------|-------|
| mAP@0.5    | 0.88  |
| Precision  | 0.82  |
| Recall     | 0.83  |
| F1 Score   | 0.59  |

- Loss curves and PR curves show steady improvement across training versions.
- Confusion matrices confirm balanced and accurate classification.

---

## 💻 Tech Stack

| Layer            | Technology                      |
|------------------|----------------------------------|
| ML Model         | YOLOv8 (Ultralytics)             |
| Language         | Python 3.10                      |
| Backend          | Flask                            |
| Frontend         | HTML, CSS, JavaScript            |
| Media Processing | OpenCV                           |
| Training         | Google Colab (GPU)               |
| Annotation       | Roboflow                         |
| Optional Storage | SQLite / CSV                     |

---

## 📂 Folder Structure

```bash
├── app/
│   ├── static/
│   ├── templates/
│   ├── model/
│   ├── utils/
│   └── app.py
├── models/
│   ├── crash_detector.pt
│   └── fall_detector.pt
├── data/
│   └── sample_inputs/
├── results/
│   └── annotated_outputs/
├── README.md
├── requirements.txt
└── run.sh
