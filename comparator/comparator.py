from keras.layers import Input, Dense, concatenate, Lambda
from keras.models import Model
from keras import backend as K


def enum(**enums):
    return type('Enum', (), enums)


class Comparator(object):

    def __init__(self, input_size=128, loss=0):
        self.__model = None
        self.__input_shape = (1, input_size)

    def __input_layer(self, name='input'):
        return Input(shape=self.__input_shape, name=name)

    @staticmethod
    def __fully_connected_layer(layer, neurons=1024, name='dense', activation='relu'):
        return Dense(units=neurons, activation=activation, name=name)(layer)

    def get_model(self):
        if self.__model is None:
            self.__call__()
        return self.__model

    def __call__(self, *args, **kwargs):
        input_reference = self.__input_layer('reference_layer')
        input_test = self.__input_layer('test_layer')
        ref_fc_layer = Comparator.__fully_connected_layer(input_reference, neurons=1024, name='ref_dense_layer',
                                                          activation='relu')
        test_fc_layer = Comparator.__fully_connected_layer(input_test, neurons=1024, name='test_dense_layer',
                                                           activation='relu')
        comparator_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        vector_distance = comparator_layer([ref_fc_layer, test_fc_layer])
        prediction = Dense(units=1, activation='sigmoid')(vector_distance)
        self.__model = Model(input=[input_reference, input_test], outputs=prediction)
        return self.get_model()
