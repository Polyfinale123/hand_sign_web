from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque

app = Flask(__name__)

# Load model
model_dict = pickle.load(open("model_videos.p", 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Labels
labels_dict = {0: 'Hii', 1: 'I am SLD', 2: 'Nice to meet you', 3: 'How Are You', 4: 'Feeling Better'}
frame_buffer = deque(maxlen=20)
conversation_log = []

prev_character = None
start_time = None
spoken = False

def generate_frames():
    global prev_character, start_time, spoken, frame_buffer

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        box_x1, box_y1 = W // 4, H // 4
        box_x2, box_y2 = (W // 4) * 3, (H // 4) * 3
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 4)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                data_aux = []
                inside_count = 0
                total_landmarks = 0

                for lm in hand_landmarks.landmark:
                    x_pixel = int(lm.x * W)
                    y_pixel = int(lm.y * H)
                    x_.append(lm.x)
                    y_.append(lm.y)
                    total_landmarks += 1
                    if box_x1 < x_pixel < box_x2 and box_y1 < y_pixel < box_y2:
                        inside_count += 1

                hand_coverage = (inside_count / total_landmarks) * 100
                if hand_coverage < 70:
                    continue

                min_x, min_y = min(x_), min(y_)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                frame_buffer.append(data_aux)

                max_features = max(len(f) for f in frame_buffer) if frame_buffer else 0
                input_data = np.array([
                    f + [0] * (max_features - len(f)) if len(f) < max_features else f
                    for f in frame_buffer
                ])

                if input_data.shape[0] >= 20:
                    input_data = input_data.flatten().reshape(1, -1)
                    if input_data.shape[1] == model.n_features_in_:
                        prediction = model.predict(input_data)
                        predicted_character = labels_dict.get(int(prediction[0]), "?")

                        if predicted_character == prev_character:
                            if start_time is None:
                                start_time = time.time()
                                spoken = False
                            elif time.time() - start_time >= 1 and not spoken:
                                conversation_log.append(predicted_character)
                                spoken = True
                        else:
                            start_time = None
                            spoken = False

                        prev_character = predicted_character

                if hand_coverage >= 70:
                    cv2.putText(frame, prev_character if prev_character else "?", 
                                (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.3, (0, 255, 0), 3, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', logs=conversation_log)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def get_logs():
    return jsonify(conversation_log)

if __name__ == '__main__':
    app.run(debug=True)
