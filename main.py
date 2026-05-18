import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import winsound   # Windows beep sound

MODEL_PATH = "best.pt"
VIDEO_PATH = "08.mp4"   # Change to 0 for webcam

QUEUE_LENGTH = 10
CONFIDENCE_THRESHOLD = 0.4

model = YOLO(MODEL_PATH)


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

    bottom_left = [0, rows]
    top_left = [cols * 0.4, rows * 0.6]
    top_right = [cols * 0.6, rows * 0.6]
    bottom_right = [cols, rows]

    vertices = np.array(
        [[bottom_left, top_left, top_right, bottom_right]],
        dtype=np.int32
    )

    return filterRegion(image, vertices)

def houghLines(image):

    return cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=300
    )


def average_slope_intercept(lines):

    left_lines = []
    left_weights = []

    right_lines = []
    right_weights = []

    if lines is None:
        return None, None

    for line in lines:

        for x1, y1, x2, y2 in line:

            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = None
    right_lane = None

    if len(left_weights) > 0:
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)

    if len(right_weights) > 0:
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)

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

    left_line = makeLinePoints(y1, y2, left_lane)
    right_line = makeLinePoints(y1, y2, right_lane)

    return left_line, right_line



def drawLaneLines(image, lines, color=(0, 255, 0), thickness=8):

    line_image = np.zeros_like(image)

    for line in lines:

        if line is not None:

            (x1, y1), (x2, y2) = line

            cv2.line(
                line_image,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)



def point_side_of_line(px, py, x1, y1, x2, y2):

    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)



class LaneObjectDetector:

    def __init__(self):

        self.last_alert_time = 0

    def process(self, frame):

        original = frame.copy()

       

        gray = convertGrayScale(frame)

        blur = applySmoothing(gray)

        edges = detectEdges(blur)

        roi = selectRegion(edges)

        lines = houghLines(roi)

        leftLine, rightLine = laneLines(frame, lines)

        lane_frame = drawLaneLines(
            frame,
            (leftLine, rightLine)
        )

       

        results = model.predict(
            lane_frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])

                cls = int(box.cls[0])

                label = model.names[cls]

                # Draw bounding box
                cv2.rectangle(
                    lane_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                text = f"{label} {conf:.2f}"

                cv2.putText(
                    lane_frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(
                    lane_frame,
                    (cx, cy),
                    5,
                    (255, 0, 0),
                    -1
                )

                

                crossed = False

                # LEFT LINE
                if leftLine is not None:

                    (lx1, ly1), (lx2, ly2) = leftLine

                    side = point_side_of_line(
                        cx, cy,
                        lx1, ly1,
                        lx2, ly2
                    )

                    if side > 0:
                        crossed = True

                # RIGHT LINE
                if rightLine is not None:

                    (rx1, ry1), (rx2, ry2) = rightLine

                    side = point_side_of_line(
                        cx, cy,
                        rx1, ry1,
                        rx2, ry2
                    )

                    if side < 0:
                        crossed = True

                

                if crossed:

                    cv2.putText(
                        lane_frame,
                        "LANE CROSSING ALERT!",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        4
                    )

                    current_time = time.time()

                    # Beep every 1 second
                    if current_time - self.last_alert_time > 1:

                        self.last_alert_time = current_time

                        try:
                            winsound.Beep(2000, 500)
                        except:
                            print("ALERT!")

        return lane_frame



if __name__ == "__main__":

    detector = LaneObjectDetector()

    # Webcam
    # cap = cv2.VideoCapture(0)

    # Video file
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        result = detector.process(frame)

        cv2.imshow("Lane + Object Detection", result)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()