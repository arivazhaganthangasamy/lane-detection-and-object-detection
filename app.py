from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import os
import json
import time
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__)
app.secret_key = 'lane_detection_secret_key_2024'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Simple in-memory user store (replace with DB in production)
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ── Lane Detection Core ──────────────────────────────────────────────────────

def applySmoothing(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def convertGrayScale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detectEdges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def filterRegion(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
    return cv2.bitwise_and(image, mask)

def selectRegion(image):
    rows, cols = image.shape[:2]
    bottom_left  = [0, rows]
    top_left     = [cols * 0.4, rows * 0.6]
    top_right    = [cols * 0.6, rows * 0.6]
    bottom_right = [cols, rows]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filterRegion(image, vertices)

def houghLines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180,
                           threshold=20, minLineLength=20, maxLineGap=300)

def average_slope_intercept(lines):
    left_lines, left_weights   = [], []
    right_lines, right_weights = [], []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2-y1)**2 + (x2-x1)**2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    left_lane  = np.dot(left_weights,  left_lines)  / np.sum(left_weights)  if left_weights  else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def makeLinePoints(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    if slope == 0:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))

def laneLines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    return makeLinePoints(y1, y2, left_lane), makeLinePoints(y1, y2, right_lane)

def drawLaneLines(image, lines, color=(0, 255, 0), thickness=8):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            (x1, y1), (x2, y2) = line
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def point_side_of_line(px, py, x1, y1, x2, y2):
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

def process_frame(frame, model=None, conf_threshold=0.4):
    gray   = convertGrayScale(frame)
    blur   = applySmoothing(gray)
    edges  = detectEdges(blur)
    roi    = selectRegion(edges)
    lines  = houghLines(roi)
    leftLine, rightLine = laneLines(frame, lines)
    lane_frame = drawLaneLines(frame, (leftLine, rightLine))
    alert = False

    if model is not None:
        results = model.predict(lane_frame, conf=conf_threshold, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                label = model.names[cls]
                cv2.rectangle(lane_frame, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(lane_frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                cv2.circle(lane_frame, (cx, cy), 5, (255,0,0), -1)
                crossed = False
                if leftLine:
                    (lx1,ly1),(lx2,ly2) = leftLine
                    if point_side_of_line(cx,cy,lx1,ly1,lx2,ly2) > 0:
                        crossed = True
                if rightLine:
                    (rx1,ry1),(rx2,ry2) = rightLine
                    if point_side_of_line(cx,cy,rx1,ry1,rx2,ry2) < 0:
                        crossed = True
                if crossed:
                    alert = True
                    cv2.putText(lane_frame, "LANE CROSSING ALERT!",
                                (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0,0,255), 4)
    return lane_frame, alert

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('detection'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        users = load_users()
        if username in users and check_password_hash(users[username]['password'], password):
            session['user'] = username
            return redirect(url_for('detection'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')
        email    = request.form.get('email', '').strip()
        if not username or not password or not email:
            error = 'All fields are required.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        else:
            users = load_users()
            if username in users:
                error = 'Username already exists.'
            else:
                users[username] = {
                    'password': generate_password_hash(password),
                    'email': email,
                    'created': time.strftime('%Y-%m-%d')
                }
                save_users(users)
                session['user'] = username
                return redirect(url_for('detection'))
    return render_template('signup.html', error=error)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html', user=session.get('user'))

@app.route('/detection')
@login_required
def detection():
    return render_template('detection.html', user=session.get('user'))

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return jsonify({'success': True, 'filename': filename})

@app.route('/process_video/<filename>')
@login_required
def process_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    def generate():
        model = None
        try:
            from ultralytics import YOLO
            if os.path.exists('best.pt'):
                model = YOLO('best.pt')
        except:
            pass

        cap = cv2.VideoCapture(filepath)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 360))
            processed, _ = process_frame(frame, model)
            _, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_image', methods=['POST'])
@login_required
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    img_array = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    model = None
    try:
        from ultralytics import YOLO
        if os.path.exists('best.pt'):
            model = YOLO('best.pt')
    except:
        pass

    processed, alert = process_frame(frame, model)
    _, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 90])
    import base64
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': img_b64, 'alert': alert})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=False,host='0.0.0.0',port=5000)
