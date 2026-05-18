# LaneVision AI — Flask Web App

Open CV lane detection + YOLOv11 object detection served as a full web application.

## Project Structure
```
lane_detect_app/
├── app.py                  ← Flask backend
├── requirements.txt
├── users.json              ← Auto-created on first signup
├── uploads/                ← Uploaded media stored here
├── best.pt                 ← Place your YOLOv11 weights here
└── templates/
    ├── base.html           ← Shared nav + styles
    ├── login.html          ← Sign-in page
    ├── signup.html         ← Registration page
    ├── about.html          ← About / landing page
    └── detection.html      ← Detection dashboard
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy your YOLOv11 weights
cp /path/to/best.pt .

# 3. Run the app
python app.py
```

Visit **http://localhost:5000** — you'll be redirected to the login page.

## Features
- ✅ Signup / Login with hashed passwords (werkzeug)
- ✅ opencv(Canny + Hough lines)
- ✅ YOLOv11 object detection via Ultralytics
- ✅ Lane crossing alert logic
- ✅ Live MJPEG video streaming endpoint (`/process_video/<filename>`)
- ✅ Single-image processing endpoint (`/process_image`)
- ✅ Confidence threshold control
- ✅ Session statistics + event log
- ✅ Drag-and-drop file upload

## Running without best.pt
The app works fine without `best.pt` — lane lines will still be drawn but no bounding boxes will appear. Place `best.pt` in the same directory as `app.py` to enable object detection.

## Production Notes
- Replace the in-memory `users.json` store with SQLite / PostgreSQL for production.
- Set a strong `SECRET_KEY` in `app.py`.
- Use gunicorn: `gunicorn -w 1 app:app` (single worker required for video streaming).

If you would like to check the output of this project, please click the link below.
https://lane-detection-and-object-detection-1.onrender.com
After opening the link, create an account and log in.
Then upload the video file.
The processing may take a few minutes after the upload is completed.
