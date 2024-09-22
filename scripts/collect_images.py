import os.path

import cv2


class CollectImages:
    def __init__(self, num_of_classes, dataset_size):
        self.num_of_classes = num_of_classes
        self.dataset_size = dataset_size
        self.data_directory = '../data'

    def main(self):
        self.create_folders()
        self.collect_images()

    def create_folders(self):
        if not os.path.exists(self.data_directory):
            os.mkdir(self.data_directory)

        for i in range(self.num_of_classes):
            if not os.path.exists(f'{self.data_directory}/{i}'):
                os.mkdir(f'{self.data_directory}/{i}')


    def collect_images(self):
        cap = cv2.VideoCapture(0)

        for i in range(self.num_of_classes):
            print(f'Collecting data for class {i} / {chr(97 + i)}')

            while True:
                ret, frame = cap.read()
                cv2.putText(
                    frame,
                    'Press Q to start',
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 255, 0),
                    3, cv2.LINE_AA
                )

                cv2.imshow('frame', frame)

                if cv2.waitKey(25) == ord('q'):
                    break

            counter = 0
            while counter < self.dataset_size:
                ret, frame = cap.read()
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                cv2.imwrite(f'{self.data_directory}/{i}/{counter}.jpg', frame)

                counter += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    CollectImages(26, 100).main()

