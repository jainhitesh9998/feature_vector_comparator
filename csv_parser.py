import csv
import os
import numpy as np
from feature_vector.feature_vector import FeatureVector
def read_csv():
    fv = FeatureVector()
    path = "/home/developer/Desktop/reid/market1501/dataset.csv"
    print(os.path.isfile(path))
    with open(path, 'r') as dataset:
        data = csv.reader(dataset)
        print()
        for count, _ in enumerate(data):
            print(count)
            label = _[0]
            vec = np.asarray(_[2:])
            fv.add_vector(label, vec)
    # print(len(fv))
    return fv

if __name__ == '__main__':
    read_csv()
