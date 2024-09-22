import os
import  pickle
import cv2
import mediapipe as mp

class CreateDataSet:
    def __init__(self, num_of_classes, dataset_size):
        self.num_of_classes = num_of_classes
        self.dataset_size = dataset_size
        self.data_directory = '../data'

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    def main(self):

        data = []
        labels = []

        for d in os.listdir(self.data_directory):
            for img_path in os.listdir(f'{self.data_directory}/{d}'):
                data_aux = []
                x_ = []
                y_ = []

                img = cv2.imread(f'{self.data_directory}/{d}/{img_path}')
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
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

                    data.append(data_aux)
                    labels.append(d)

        with open(f'{self.data_directory}/data.pickle', 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)


if __name__ == '__main__':
    CreateDataSet(26, 100).main()