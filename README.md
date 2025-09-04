import cv2
import mediapipe as mp
import os

# ===== Create Save Directories =====
save_faces = "saved_faces"
save_left_hands = "saved_left_hands"
save_right_hands = "saved_right_hands"
save_background = "saved_background"

for folder in [save_faces, save_left_hands, save_right_hands, save_background]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ===== Initialize =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

# ===== Counters =====
face_count = 0
left_hand_count = 0
right_hand_count = 0
background_count = 0

print("✅ Live detection started. Press 'q' to quit.")

finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
finger_tips = [4, 8, 12, 16, 20]  # Mediapipe landmark indices for fingertips

# ===== Helper: Check if finger is up =====
def fingers_up(hand_landmarks):
    fingers_status = []
    # Thumb
    fingers_status.append(
        hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
        if hand_landmarks.landmark[17].x < hand_landmarks.landmark[5].x
        else hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
    )
    # Other fingers
    for tip_id in [8, 12, 16, 20]:
        fingers_status.append(
            hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y
        )
    return fingers_status

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ===== Detect Hands =====
    results = hands.process(rgb_frame)
    detected_hands = []

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            hand_label = hand_handedness.classification[0].label  # Left or Right
            detected_hands.append(hand_label)

            # Draw landmarks on live frame only (for display)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingers up
            fingers_status = fingers_up(hand_landmarks)

            # Show Finger Names only when raised
            h, w, _ = frame.shape
            for tip_index, name, is_up in zip(finger_tips, finger_names, fingers_status):
                if is_up:
                    cx = int(hand_landmarks.landmark[tip_index].x * w)
                    cy = int(hand_landmarks.landmark[tip_index].y * h)
                    cv2.putText(
                        frame,
                        f"{name}",
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

            # Determine gesture
            if all(fingers_status):
                gesture = "Open Hand"
            elif not any(fingers_status):
                gesture = "Closed Hand"
            else:
                gesture = "Partial"

            cv2.putText(
                frame,
                f"{hand_label}: {gesture}",
                (10, 30 + 20 * detected_hands.index(hand_label)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # === Crop hand for saving WITHOUT landmarks ===
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, cx), min(y_min, cy)
                x_max, y_max = max(x_max, cx), max(y_max, cy)

            clean_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            hand_crop = clean_frame[y_min:y_max, x_min:x_max].copy()

            if hand_crop.size != 0:
                if hand_label == "Left":
                    left_hand_count += 1
                    cv2.imwrite(
                        os.path.join(
                            save_left_hands, f"left_hand_{left_hand_count}.jpg"
                        ),
                        hand_crop,
                    )
                else:
                    right_hand_count += 1
                    cv2.imwrite(
                        os.path.join(
                            save_right_hands, f"right_hand_{right_hand_count}.jpg"
                        ),
                        hand_crop,
                    )

    # ===== Detect Faces =====
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        for (x, y, w_f, h_f) in faces:
            # Draw rectangle on live frame
            cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), (0, 255, 0), 2)

            # Crop face for saving WITHOUT rectangle
            clean_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            face_crop = clean_frame[y:y + h_f, x:x + w_f].copy()
            if face_crop.size != 0:
                face_count += 1
                cv2.imwrite(
                    os.path.join(save_faces, f"face_{face_count}.jpg"), face_crop
                )

    # ===== Save full background frame =====
    if len(faces) == 0 and len(detected_hands) == 0:
        background_count += 1
        cv2.imwrite(
            os.path.join(save_background, f"background_{background_count}.jpg"),
            frame.copy(),
        )

    # ===== STATUS LOGIC =====
    num_hands = len(detected_hands)
    face_detected = len(faces) > 0

    if num_hands == 0 and not face_detected:
        status = "Background"
    elif num_hands == 1 and not face_detected:
        status = f"1 Hand ({detected_hands[0]})"
    elif num_hands == 2 and not face_detected:
        status = "2 Hands"
    elif num_hands == 0 and face_detected:
        status = "Face"
    elif num_hands == 1 and face_detected:
        status = f"1 Hand ({detected_hands[0]}) + Face"
    elif num_hands == 2 and face_detected:
        status = "2 Hands + Face"
    else:
        status = "Unknown"

    # ===== Show Status + Counters on live frame =====
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Faces Saved: {face_count}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Left Hands Saved: {left_hand_count}",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Right Hands Saved: {right_hand_count}",
        (20, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Backgrounds Saved: {background_count}",
        (20, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection ended.")
 








