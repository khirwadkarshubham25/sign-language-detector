import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


class TrainModel:
    def __init__(self):
        self.data_dir = '../data'
        self.data_file = f'{self.data_dir}/data.pickle'

    def main(self):
        data_dict = pickle.load(open(self.data_file, 'rb'))

        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        model = RandomForestClassifier()
        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)

        score = accuracy_score(y_predict, y_test)

        print(f'{score * 100}% of samples were classified correctly !')

        with open(f'{self.data_dir}/model.p', 'wb') as f:
            pickle.dump({'model': model}, f)

if __name__ == '__main__':
    TrainModel().main()