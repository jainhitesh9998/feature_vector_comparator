import cv2
import glob
from random import randint
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import os
import random
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

def generator(true_sample, false_sample,feature_vector,  batch_size=16):
    # num_samples = len(samples)
    # print(len(samples))
    true_pair = []
    false_pair = []
    # print(len(true_sample))
    # print(len(false_sample))
    while True:
        sklearn.utils.shuffle(true_sample)
        sklearn.utils.shuffle(false_sample)
        # print(true_sample)
        # print(len(true_sample))
        # print(false_sample)
        # print(len(false_sample))
        samples = true_sample + false_sample
        # print(samples)
        num_samples = len(samples)
        # print("number of samples", num_samples)
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = permute1(samples[offset: offset+batch_size], feature_vector.get_vector_dict())
            # print((batch_samples))
            batch_samples = sklearn.utils.shuffle(batch_samples)
            input1 = []
            input2 = []
            labels = []
            for batch_sample in batch_samples:
                input1.append(batch_sample[0])
                input2.append(batch_sample[1])
                labels.append([batch_sample[2]])
            # print(input.shape)
            X_train1 = np.array(input1)
            X_train2 = np.array(input2)
            Y_train = np.array(labels)
            # print("shape",X_train1.shape)
            # print("shape",X_train2.shape)
            # print("shape",Y_train.shape)

            # yield [np.expand_dims(X_train1, 0), np.expand_dims(X_train2, 0)], np.expand_dims(Y_train, 0)
            yield [np.array(X_train1), np.array(X_train2)], [np.array(Y_train)]

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

def permute(key_value, vector):
    global count_0
    global count_1
    # print(key_value)
    samples = []

    for k1, k2 in key_value:
        # print("keys ", k1, k2)
        # print("len of k1", len(vector[k1]))
        # print("len of k2", len(vector[k2]))
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
    # print("len of samples", len(samples))
    return samples

def create_pair(vector):
    global count_0
    global count_1
    keys = vector.keys()
    key_pairs = [ _ for _ in itertools.combinations_with_replacement(keys, 2)]
    # print(key_pairs)
    return key_pairs

# def extract_features(patch, ip, op):
#     patch[0] = cv2.equalizeHist(patch[0])
#     patch[1] = cv2.equalizeHist(patch[1])
#     patch[2] = cv2.equalizeHist(patch[2])
#     ip.push(Inference(patch, meta_dict={}))
#     op.wait()
#     ret, feature_inference = op.pull()
#     if ret:
#         return feature_inference.get_result()


# if __name__ == '__main__':
def train():

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = True # GPU for multiple processes
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    feature_vector = make_features("/home/developer/Desktop/mars_dataset.csv")
    # print(feature_vector.get_vector_dict().keys())
    # print(feature_vector.get_vector_dict()['741'])
    # key_pair = create_pair(feature_vector.get_vector_dict())
    # print("no. of keys", len(feature_vector.get_vector_dict().keys()))
    # return
    # print(count_0)
    # print(count_1)
    # print(feature_vector.get_vector_dict())
    true_sample = []
    false_sample = []
    # d = Distance.ABSOLUTE
    model = Comparator(input_size=128, distance=Distance.EUCLIDIAN, hidden_layers=2)()
    # sklearn.utils.shuffle(key_pair)
    # for kp in key_pair:
    #     if (kp[0] == kp[1]):
    #         true_sample.append(kp)
    #     else:
    #         false_sample.append(kp)
    # print("len of true sample", len(true_sample))
    # print("len of false sample", len(false_sample))

    all_classes = list(feature_vector.get_vector_dict().keys())
    # print(all_classes)
    num_of_classes  = len(all_classes)

    for tr in all_classes:
        tr_len = len(feature_vector.get_vector_dict()[tr])
        for i in range(tr_len):
            for j in range(tr_len):
                true_sample.append([(tr, i), (tr, j)])

    # print(true_sample[:50])
    true_samples_len = len(true_sample)
    false_per_class = int(true_samples_len/num_of_classes * 1.5)

    for tr in all_classes:
        for _ in range(false_per_class):
            ref_class_image = random.randint(0,len(feature_vector.get_vector_dict()[tr])-1)
            random_class = random.choice(all_classes)
            if random_class!=tr:
                random_class_image = random.randint(0,len(feature_vector.get_vector_dict()[random_class])-1)
                false_sample.append([(tr, ref_class_image), (random_class, random_class_image)])

    false_samples_len = len(false_sample)
    # print(len(all_classes))
    # print(true_samples_len)
    # print(false_per_class)
    # print(false_samples_len)
    # print(false_sample[:50])

    # print()
    # print(samples[1])
    # print(len(samples))
    train_true_samples, val_true_samples = train_test_split(true_sample, test_size=0.2)
    train_false_samples, val_false_samples = train_test_split(false_sample, test_size=0.2)
    train_generator = generator(train_true_samples, train_false_samples, feature_vector, batch_size=32)
    validation_generator = generator(val_true_samples,val_false_samples, feature_vector, batch_size=32)
    epoch = 10
    saved_weights_name = 'model_' '' + time.strftime("%m_%d_%Y_%H_%M_%S") +'.h5'
    print(saved_weights_name)
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.001,
                               patience=3,
                               mode='min',
                               verbose=1)
    checkpoint = ModelCheckpoint(saved_weights_name,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)
    tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=False)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae', 'acc'])
    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_true_samples)*2,
                        epochs=epoch,
                        verbose=1,
                        validation_data = validation_generator,
                        nb_val_samples = len(val_true_samples)*2,
                        callbacks=[early_stop, checkpoint, tensorboard]
                        )
    # model.save_weights('new_weights.h5')

def test():
    model = Comparator()()
    model.load_weights('model_.h5')
    model.summary()
    feature_vector = FeatureVector()
    session_runner = SessionRunner()
    extractor = MarsExtractorAPI('mars_api', True)
    ip = extractor.get_in_pipe()
    op = extractor.get_out_pipe()
    extractor.use_session_runner(session_runner)
    session_runner.start()
    extractor.run()
    image_files = []
    for id in range(1, 5):
        image_files.append(glob.glob(
            '/home/allahbaksh/Tailgating_detection/SecureIt/data/obj_tracking/outputs/patches/{}/*.jpg'.format(id)))
    # print(len(image_files))
    patch0 = cv2.imread(image_files[0][randint(0, len(image_files[0]))])
    patch0_1 = cv2.imread(image_files[0][randint(0, len(image_files[0]))])
    patch1 = cv2.imread(image_files[1][randint(0, len(image_files[1]))])
    patch2 = cv2.imread(image_files[2][randint(0, len(image_files[2]))])
    patch3 = cv2.imread(image_files[3][randint(0, len(image_files[3]))])
    f_vec0 = np.array(extract_features(patch0, ip, op))
    f_vec0_1 = np.array(extract_features(patch0_1, ip, op))
    f_vec1 = np.array(extract_features(patch1, ip, op))
    f_vec2 = np.array(extract_features(patch2, ip, op))
    f_vec3 = np.array(extract_features(patch3, ip, op))
    #print(f_vec1)

    output = model.predict([np.expand_dims(f_vec0, 0), np.expand_dims(f_vec0_1, 0)])
    # print(output)

if __name__ == '__main__':
    train()
