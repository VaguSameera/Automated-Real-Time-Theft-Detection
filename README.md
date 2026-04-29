# Automated-Real-Time-Theft-Detection
A real-time automated retail theft detection system using YOLOv8 and deep learning to identify shoplifting behavior from CCTV footage. Integrates OpenCV, Flask, and a web dashboard for live monitoring, alerts, and detection history

PROJECT STRUCTURE 
├── app.py                # Flask backend
├── detect.py             # YOLO model logic
├── static/               # CSS, JS, assets
├── templates/            # HTML pages
├── dataset/              # Training data
├── models/               # Trained weights
├── uploads/              # Input videos
└── database/             # SQLite DB

# Clone repository
git clone https://github.com/your-username/retail-theft-detection.git
cd retail-theft-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Tech Stack:
Programming Language: Python
Framework: Flask
Deep Learning: YOLOv8 (Ultralytics), PyTorch
Computer Vision: OpenCV
Frontend: HTML, CSS, JavaScript
Database: SQLite
