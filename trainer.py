from random import randint

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from comparator.comparator import Comparator
from feature_vector.feature_vector import FeatureVector
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import  os
from collections import OrderedDict
import itertools
from utils import utils

count_0 = 0
count_1 = 0
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
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
            # print(X_train2.shape)
            # print(Y_train.shape)

            yield [np.expand_dims(X_train1, 0), np.expand_dims(X_train2, 0)], np.expand_dims(Y_train, 0)

def create_samples(vector):
    global count_0
    global count_1
    keys = vector.keys()
    key_pairs = [ _ for _ in itertools.combinations_with_replacement(keys, 2)]
    print(key_pairs)
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


if __name__ == '__main__':
    #data = utils.read_csv('extras/data.csv')
    feature_vector = FeatureVector()
    #for idx, vec in data:
    #    feature_vector.add_vector(idx, vec)
    for x in range(200):
        feature_vector.add_vector(randint(0, 30), [randint(0, 128) for _ in range(128)])
    samples = create_samples(feature_vector.get_vector_dict())
    print(count_0)
    print(count_1)
    # print(feature_vector.get_vector_dict())
    model = Comparator()()
    sklearn.utils.shuffle(samples)
    # print()
    # print(samples[1])
    # print(len(samples))
    train_samples, val_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples)
    validation_generator = generator(val_samples)
    epoch = 10
    saved_weights_name = './model.h5'
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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae', 'acc'])
    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_samples),
                        epochs=epoch,
                        verbose=1,
                        validation_data = validation_generator,
                        nb_val_samples = len(val_samples),
                        callbacks=[early_stop, checkpoint, tensorboard]
                        )





