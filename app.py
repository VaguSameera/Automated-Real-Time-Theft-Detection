from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
from ultralytics import YOLO
import numpy as np
import imutils
import cv2
import os
import time
import uuid
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Import parameters
from config.parameters import WIDTH, start_status, shoplifting_status, not_shoplifting_status
from config.parameters import cls0_rect_color, cls1_rect_color, conf_color, status_color

app = Flask(__name__)
app.secret_key = "retail_theft_secret_key"   # Required for sessions

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
DETECTIONS_FOLDER = 'static/detections'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

current_video_path = None 
mymodel = YOLO(r"configs\shoplifting_wights.pt")

# Global storage for detections
detection_history = []


# =========================
# LOGIN SYSTEM
# =========================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        cursor.execute("SELECT id, password FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session["logged_in"] = True
            session["user_id"] = user[0]
            return redirect(url_for('index'))
        else:
            return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (email, password) VALUES (?, ?)",
                (email, hashed_password)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error="Email already registered")

        conn.close()

# Auto login after registration
        session["logged_in"] = True
        return redirect(url_for("index"))

    return render_template('register.html')


# =========================
# PROTECTED ROUTES
# =========================

def login_required():
    return not session.get("logged_in")


@app.route('/', methods=['GET', 'POST'])
def index():
    if login_required():
        return redirect(url_for("login"))

    global current_video_path
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            current_video_path = filepath
            return render_template('home.html', active_page='home', video_ready=True, filename=filename)

    return render_template('home.html', active_page='home', video_ready=False, filename="NO INPUT")


@app.route('/detect')
def detect_page():
    if login_required():
        return redirect(url_for("login"))
    return render_template('detect.html', active_page='detect')


@app.route('/team')
def team():
    if login_required():
        return redirect(url_for("login"))
    return render_template('team.html', active_page='team')


@app.route('/about')
def about():
    if login_required():
        return redirect(url_for("login"))
    return render_template('about.html', active_page='about')


@app.route('/video_feed')
def video_feed():
    if login_required():
        return redirect(url_for("login"))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_detections')
def get_detections():
    if login_required():
        return redirect(url_for("login"))

    global detection_history
    after = request.args.get('after', default=0, type=int)
    new_logs = detection_history[after:]

    return jsonify({
        "logs": new_logs,
        "total_count": len(detection_history)
    })


# =========================
# VIDEO PROCESSING FUNCTION
# =========================

def generate_frames():
    global current_video_path, detection_history
    
    if not current_video_path or not os.path.exists(current_video_path):
        return

    cap = cv2.VideoCapture(current_video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = imutils.resize(frame, width=WIDTH)
        
        result = mymodel.predict(frame)
        cc_data = np.array(result[0].boxes.data)

        status = start_status
        
        if len(cc_data) != 0:
            xywh = np.array(result[0].boxes.xywh).astype("int32")
            xyxy = np.array(result[0].boxes.xyxy).astype("int32")
            
            for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, conf, clas) in zip(xyxy, xywh, cc_data):
                person = frame[y1:y1+h, x1:x1+w]
                
                if clas == 1:
                    cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), cls1_rect_color, 2)
                    half_w = w / 2
                    half_h = h / 2
                    x = int(half_w + x1)
                    cv2.circle(frame, (x, y1), 6, (0, 0, 255), 8)

                    text = "{}%".format(np.round(conf * 100, 2))
                    cv2.putText(frame, text, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)
                    status = shoplifting_status

                    if frame_count % 30 == 0:
                        img_name = f"det_{uuid.uuid4().hex[:8]}.jpg"
                        img_path = os.path.join(DETECTIONS_FOLDER, img_name)
                        cv2.imwrite(img_path, frame) 
                        
                        detection_history.append({
                            "time": time.strftime("%H:%M:%S"),
                            "conf": int(conf * 100),
                            "image": img_name
                        })
                
                elif clas == 0 and conf > 0.8:
                    cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), cls0_rect_color, 1)
                    text = "{}%".format(np.round(conf * 100, 2))
                    cv2.putText(frame, text, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)
                    status = not_shoplifting_status

        cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        frame_count += 1
        
        (flag, encodedImage) = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not flag: 
            continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    cap.release()


if __name__ == '__main__':
    app.run(debug=True, port=5000)