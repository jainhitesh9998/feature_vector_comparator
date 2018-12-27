import cv2
import glob
from random import randint
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import os
from collections import OrderedDict
import itertools
# from feature_extraction.mars_api.mars_api import MarsExtractorAPI
from comparator.comparator import Comparator, Distance
import csv
from feature_vector.feature_vector import FeatureVector
# from tf_session.tf_session_runner import SessionRunner
# from tf_session.tf_session_utils import Inference
from utils.utils import make_features
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import random
count_0 = 0
count_1 = 0
import time

def create_samples(vector):
    global count_0
    global count_1
    keys = vector.keys()
    key_pairs = [ _ for _ in itertools.combinations_with_replacement(keys, 2)]
    # print(key_pairs)
    samples = []
    for k1, k2 in key_pairs:
        val1 = vector[k1]
        val2 = vector[k2]
        label = 0
        if k1 == k2:
            count_1+=1
            label = 1
        else:
            count_0+=1
        sample = [[v1, v2, label] for v1, v2 in itertools.product(val1, val2)]
        samples.extend(sample)
    return samples

def permute1(pairs, vector):
    samples = []
    # print("pairs")
    # print(pairs)
    for pair in pairs:
        sample = []
        sample.append(vector[pair[0][0]][pair[0][1]])
        sample.append(vector[pair[1][0]][pair[1][1]])
        if pair[0][0]==pair[1][0]:
            sample.append(1)
        else:
            sample.append(0)
        # print(sample)
        samples.append(sample)
    # print("len of samples", len(samples))
    return samples


def permute(key_value, vector):
    global count_0
    global count_1
    # print(type(key_value))
    # print(len(key_value))
    samples = []
    for k1, k2 in key_value:
        print("keys ", k1, k2)
        val1 = sklearn.utils.shuffle(vector[k1])[:15]
        val2 = sklearn.utils.shuffle(vector[k2])[:15]
        label = 0
        if k1 == k2:
            count_1 += 1
            label = 1
        else:
            count_0 += 1
        sample = [[v1, v2, label] for v1, v2 in itertools.product(val1, val2)]
        samples.extend(sample)
    return samples

def create_pair(vector):
    global count_0
    global count_1
    keys = vector.keys()
    key_pairs = [ _ for _ in itertools.combinations_with_replacement(keys, 2)]
    # print(key_pairs)
    return key_pairs

def test():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = True  # GPU for multiple processes
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)

    feature_vector = make_features("/home/developer/Desktop/mars_infy_dataset_full_image.csv")
    key_pair = create_pair(feature_vector.get_vector_dict())
    # print(count_0)
    # print(count_1)
    # print(feature_vector.get_vector_dict())
    true_sample = []
    false_sample = []
    # d = Distance.ABSOLUTE
    model = Comparator(input_size=128, distance=Distance.ABSOLUTE, hidden_layers=2)()
    model.load_weights('model_12_27_2018_11_11_39.h5')
    print("weights loaded")
    # sklearn.utils.shuffle(key_pair)
    # for kp in key_pair:
    #     if (kp[0] == kp[1]):
    #         true_sample.append(kp)
    #     else:
    #         false_sample.append(kp)
    # print("samples created")
    all_classes = list(feature_vector.get_vector_dict().keys())
    num_of_classes = len(all_classes)
    for tr in all_classes:
        tr_len = len(feature_vector.get_vector_dict()[tr])
        for i in range(tr_len):
            for j in range(tr_len):
                true_sample.append([(tr, i), (tr, j)])
    true_samples_len = len(true_sample)


    false_per_class = int(true_samples_len / num_of_classes) + 2
    for tr in all_classes:
        for _ in range(false_per_class):
            ref_class_image = random.randint(0, len(feature_vector.get_vector_dict()[tr]) - 1)
            random_class = random.choice(all_classes)
            if random_class != tr:
                random_class_image = random.randint(0, len(feature_vector.get_vector_dict()[random_class]) - 1)
                false_sample.append([(tr, ref_class_image), (random_class, random_class_image)])
    false_samples_len = len(false_sample)


    # for i in range(10):
    #     idx = random.randint(0, len(true_sample))
    #     # t_pair = permute(true_sample[idx: idx +1 ], feature_vector.get_vector_dict())
    #     t_pair = permute(false_sample[idx: idx+1], feature_vector.get_vector_dict())
    #     test_pairs = samples(t_pair)
    #     print("prediction")
    #     output = model.predict([np.array(test_pairs[0]), np.array(test_pairs[1])])
    #     print(output)
    true_08 = 0
    true_05 = 0
    false_02 = 0
    false_05 = 0
    batch_size = 1024
    for i in range(int(true_samples_len/batch_size)):
        true_sa = permute1(true_sample[i*batch_size: i*batch_size +batch_size ], feature_vector.get_vector_dict())
        test_pairs = samples(true_sa)
        # print("prediction")
        output = model.predict([np.array(test_pairs[0]), np.array(test_pairs[1])])
        greater_than_08 = len(list(filter(lambda x: x[0] >= 0.8, output)))
        greater_than_05 = len(list(filter(lambda x: x[0] >= 0.5, output)))
        print(greater_than_05),
        print("/1024 for 0.5")
        print(greater_than_08),
        print("/1024 for 0.8")
        true_08 += greater_than_08
        true_05 += greater_than_05


    for i in range(int(false_samples_len/batch_size)):
        false_sa = permute1(false_sample[i*batch_size: i*batch_size +batch_size ], feature_vector.get_vector_dict())
        test_pairs = samples(false_sa)
        # print("prediction")
        output = model.predict([np.array(test_pairs[0]), np.array(test_pairs[1])])
        less_than_02 = len(list(filter(lambda x: x[0] <= 0.2, output)))
        less_than_05 = len(list(filter(lambda x: x[0] < 0.5, output)))
        print(less_than_05),
        print("/1024 for 0.5")
        print(less_than_02),
        print("/1024 for 0.2")
        false_02 += less_than_02
        false_05 += less_than_05
    print(true_samples_len),
    print(true_05, )
    print(" greater than 0.5")
    print(true_08, )
    print(" greater than 0.8")

    print(true_samples_len),
    print(false_05),
    print(" less than 0.5")
    print(false_02),
    print(" less than 0.2")
    # false_sa = permute1(false_sample, feature_vector.get_vector_dict())
    # test_pairs = samples(false_sa)
    #
    # print("prediction")
    # output = model.predict([np.array(test_pairs[0]), np.array(test_pairs[1])])
    # print(type(output))
    # print(len(output))
    # return
    # print(output)

def samples(sample):
    input1 = []
    input2 = []
    labels = []
    for batch_sample in sample:
        input1.append(batch_sample[0])
        input2.append(batch_sample[1])
        labels.append([batch_sample[2]])
    X_train1 = np.array(input1)
    X_train2 = np.array(input2)
    Y_train = np.array(labels)
    return (X_train1, X_train2, Y_train)

if __name__ == '__main__':
    test()