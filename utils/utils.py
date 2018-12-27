import csv
import os
import numpy as np
from feature_vector.feature_vector import FeatureVector


def convert_int(x):
    try:
        return int(x)
    except ValueError:
        return x

def read_csv(file_name):
    output_list = []
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            _ = [convert_int(x) for x in row]
            output_list.append(_)
    return output_list

def make_features(path):
    fv = FeatureVector()
    # path = "/home/developer/Desktop/reid/market1501/dataset.csv"
    print(os.path.isfile(path))
    with open(path, 'r') as dataset:
        data = csv.reader(dataset)
        print()
        for count, _ in enumerate(data):
            # print(count)
            label = _[0]
            vec = np.asarray(_[2:])
            # print(len(vec))

            if int(label) == -1 or int(label) > 751:
                continue
            fv.add_vector(label, vec)
    # print(fv.get_vector_dict().keys())
    # print(len(fv))
    return fv