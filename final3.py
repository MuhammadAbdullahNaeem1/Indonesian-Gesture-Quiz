import pickle
import cv2
import mediapipe as mp
import numpy as np
import random
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
}


def classify_gesture(model, frame):
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1 - 10, y1 - 10),
                      (x2 + 10, y2 + 10), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        return predicted_character
    else:
        return None


def hand_gesture_game(model):
    cap = cv2.VideoCapture(0)
    score = 0
    question_count = 0
    feedback = ""
    feedback_display_time = 0
    countdown_duration = 10
    countdown = countdown_duration
    start_time = time.time()
    question_letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    with open('model.p', 'rb') as model_file:
        model = pickle.load(model_file)
    model = model_dict['model']
    while question_count < 5:
        success, img = cap.read()
        h, w, c = img.shape
        img = cv2.flip(img, 1)

        user_gesture = classify_gesture(model, img)

        elapsed_time = time.time() - start_time

        if elapsed_time > countdown_duration:
            if question_letter is None:
                question_letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

            if user_gesture == question_letter:
                feedback = "Correct"
                score += 1
            else:
                feedback = "Incorrect"

            feedback_display_time = time.time()
            start_time = time.time()
            question_count += 1
            question_letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            countdown = countdown_duration

        else:
            countdown = countdown_duration - int(elapsed_time)

            if feedback and time.time() - feedback_display_time > 3:
                feedback = ""

        feedback_x = w // 2
        feedback_y = int((h - 50 + h) / 2) + 20

        cv2.putText(img, f"Show Gesture: {question_letter}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Score: {score}/5", (w - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Countdown: {countdown}", (w - 250,
                    h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if user_gesture:
            cv2.putText(img, f"Detected: {user_gesture}", (50, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if feedback == "Correct":
            cv2.putText(img, feedback, (feedback_x - 55, feedback_y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif feedback == "Incorrect":
            cv2.putText(img, feedback, (feedback_x - 55, feedback_y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Game", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    while True:
        img[:] = 0
        cv2.putText(img, "Congrats!", (w // 8, h // 3),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 4)
        cv2.putText(img, f"Your score is:", (w // 8, h // 2),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 4)
        cv2.putText(img, f"{score}/5", (w // 8, 2 * h // 3),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 4)
        cv2.imshow("Hand Gesture Game", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


with open('model.p', 'rb') as model_file:
    model = pickle.load(model_file)['model']

hand_gesture_game(model)
