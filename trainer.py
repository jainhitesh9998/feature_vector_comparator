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


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            inputs = []
            labels = []
            for batch_sample in batch_samples:
                inputs.append([batch_sample[0], batch_sample[1]])
                labels.append(batch_sample[2])
            X_train = np.array(inputs)
            Y_train = np.array(labels)
            yield X_train, Y_train

def create_samples(vector):
    keys = vector.keys()
    key_pairs = [ _ for _ in itertools.combinations(keys, 2)]
    samples = []
    for k1, k2 in key_pairs:
        val1 = vector[k1]
        val2 = vector[k2]
        label = 0
        if k1 == k2:
            label = 1
        sample = [[v1, v2, label] for v1, v2 in itertools.product(val1, val2)]
        samples.extend(sample)
    return samples


if __name__ == '__main__':
    #data = utils.read_csv('extras/data.csv')
    feature_vector = FeatureVector()
    #for idx, vec in data:
    #    feature_vector.add_vector(idx, vec)
    feature_vector.add_vector(randint(0, 10), [randint(0, 128) for _ in range(128)])
    samples = create_samples(feature_vector.get_vector_dict())
    model = Comparator()()
    sklearn.utils.shuffle(samples)
    train_samples, test_samples = train_test_split(samples, test_size=0.2)

    train_generator = generator(train_samples)
    validation_generator = generator(test_samples)
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
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_samples),
                        epochs=epoch#,
                        # verbose=1,
                        # validation_data=validation_generator,
                        # validation_steps=len(test_samples),
                        # callbacks=[early_stop, checkpoint, tensorboard]
                        )





