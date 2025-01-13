import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time

# Load YOLO model for object detection
yolo_model = YOLO("yolov8n.pt")  # Pretrained model

# Transformation matrix from Kalibr (example values)
T_head_to_arm = np.array([
    [0.9998, -0.0176, 0.0123, 0.1],  # Rotation + Translation
    [0.0176, 0.9998, -0.0056, 0.2],
    [-0.0123, 0.0056, 0.9999, 0.3],
    [0.0, 0.0, 0.0, 1.0]
])

def capture_frame(camera_id):
    """Captures a frame from the specified camera."""
    cap = cv2.VideoCapture(camera_id)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to capture frame from camera {camera_id}")
    return frame

def detect_objects(image):
    """Detects objects in the given image using YOLO."""
    results = yolo_model(image)
    detections = results.xyxy[0]  # Extract detections
    objects = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = yolo_model.names[int(cls)]
        objects.append({
            "label": label,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": float(conf)
        })
    return objects

def combine_camera_views(head_frame, arm_frame):
    """Transforms the arm camera view to align with the head camera view."""
    arm_transformed = cv2.warpPerspective(arm_frame, T_head_to_arm[:3, :], (head_frame.shape[1], head_frame.shape[0]))
    combined_view = cv2.addWeighted(head_frame, 0.5, arm_transformed, 0.5, 0)
    return combined_view

def analyze_bin():
    """Analyzes the bin using combined camera views."""
    head_frame = capture_frame(0)  # Head-mounted camera (example ID)
    arm_frame = capture_frame(1)  # Arm-mounted camera (example ID)

    # Combine views
    combined_view = combine_camera_views(head_frame, arm_frame)

    # Detect objects
    detected_objects = detect_objects(combined_view)

    # Visualize detections
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["label"]
        conf = obj["confidence"]
        cv2.rectangle(combined_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(combined_view, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Bin Analysis", combined_view)
    cv2.waitKey(5000)  # Display for 5 seconds
    cv2.destroyAllWindows()

    return detected_objects

def execute_bin_analysis():
    """Executes the bin analysis and reports findings."""
    print("Analyzing the bin...")
    detected_objects = analyze_bin()

    if any(obj["label"] == "gauze bandages" for obj in detected_objects):
        print("Gauze bandages found in the bin. Proceeding to pick up.")
    else:
        print("Gauze bandages not found in the bin. Alerting staff.")
        activate_led()
        report_issue(time.strftime("%I:%M %p"), "Gauze bandages not found in bin", "Analyze bin for gauze bandages")

# Activate LED and report issue (reuse existing functions)
def activate_led():
    """Simulates activating a large LED to alert staff."""
    print("[LED Activated] Large LED on top is lit to alert staff.")
    time.sleep(1)  # Simulate LED activation time

def report_issue(time_of_request, issue, instruction):
    """Reports the issue verbally and logs it."""
    message = (
        f"At {time_of_request}, I was asked to {instruction}. "
        f"However, {issue}."
    )
    print(f"[Robot Report] {message}")
    tts_engine = pyttsx3.init()
    tts_engine.say(message)
    tts_engine.runAndWait()

# Main execution
execute_bin_analysis()
