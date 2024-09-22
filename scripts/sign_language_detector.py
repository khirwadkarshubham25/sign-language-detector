import pickle

import cv2
import mediapipe as mp
import numpy as np


class SignLanguageDetector:
    def __init__(self):
        self.model = pickle.load(open('../data/model.p', 'rb'))['model']
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    def main(self):
        cap = cv2.VideoCapture(0)
        while True:
            data_aux, _x, _y = [], [], []
            ret, frame = cap.read()
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        self.mp_hands.HAND_CONNECTIONS,  # hand connections
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        _x.append(x)
                        _y.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(_x))
                        data_aux.append(y - min(_y))

                x1 = int(min(_x) * W) - 10
                y1 = int(min(_y) * H) - 10

                x2 = int(max(_x) * W) - 10
                y2 = int(max(_y) * H) - 10

                prediction = self.model.predict([np.asarray(data_aux)])

                predicted_character = chr(int(prediction[0]) + 65)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

            cv2.imshow('frame', frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    SignLanguageDetector().main()