import csv
import numpy as np
from sklearn.datasets._base import Bunch
from data_provider import get_matrix

def load_my_fancy_dataset():
    with open(r'C:\Users\lzx\Desktop\Imbalanced_SADs_classification\data\AT\BZH0.5_new.csv') as csv_file:
        matrix = get_matrix.csv_to_Matrix(r'C:\Users\lzx\Desktop\Imbalanced_SADs_classification\data\AT\BZH0.5_new.csv')
        # Please replace with your file path
        data_file = csv.reader(csv_file)
        # matrix = get_matrix(file_path)
        n_samples = int(matrix.shape[0])
        n_features = int(matrix.shape[1] - 2)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        count = 0
        cpue = np.empty((n_samples,), dtype=np.int)
        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[1:-1], dtype=np.float64)
            target[i] = np.asarray(sample[-1], dtype=np.int)
            cpue[i] = np.asarray(sample[0], dtype=np.float64)

    return cpue, Bunch(data=data, target=target)