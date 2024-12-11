from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import threading

app = Flask(__name__)
 
# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Shared resources
frame_lock = threading.Lock()
shared_frame = None
shared_count = 0

def calculate_angle(landmark1, landmark2):
    return np.abs(landmark1.x - landmark2.x)

def count_fingers(hand_landmarks, hand):
    finger_tips_ids = [4, 8, 12, 16, 20]
    fingers_count = 0
    if hand == 'Left':
        if calculate_angle(hand_landmarks.landmark[4], hand_landmarks.landmark[3]) > 0.04:
            fingers_count += 1
    else:
        if calculate_angle(hand_landmarks.landmark[4], hand_landmarks.landmark[3]) > 0.04:
            fingers_count += 1
    for tip_id in finger_tips_ids[1:]:
        pip_id = tip_id - 2
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y:
            fingers_count += 1
    return fingers_count

def capture_and_process_frames():
    global shared_frame, shared_count, frame_lock
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            finger_count = 0
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hand_label = handedness.classification[0].label
                    finger_count = count_fingers(hand_landmarks, hand_label)
            with frame_lock:
                shared_frame = frame.copy()
                shared_count = finger_count
    cap.release()

def generate_video_stream():
    global shared_frame, frame_lock
    while True:
        with frame_lock:
            if shared_frame is not None:
                flag, encodedImage = cv2.imencode('.jpg', shared_frame)
                if flag:
                    yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                           bytearray(encodedImage) + b'\r\n')

def generate_finger_count():
    global shared_count
    while True:
        yield f"data: {shared_count}\n\n"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count')
def count():
    return Response(generate_finger_count(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Start the frame capture thread
    thread = threading.Thread(target=capture_and_process_frames)
    thread.daemon = True
    thread.start()
    app.run(debug=True, threaded=True, use_reloader=False)
