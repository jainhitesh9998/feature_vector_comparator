from keras.layers import Input, Dense, concatenate, Lambda
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
from enum import Enum

# def enum(**enums):
#     return type('Enum', (), enums)


def euclidian_distance():
    return Lambda(lambda tensors: K.abs(K.square(tensors[0]) - K.square(tensors[1])))

def cosine_distance():
    return Lambda(
        lambda tensors: K.abs((K.dot(tensors[0], tensors[1])) / (K.square(tensors[0]) * K.square(tensors[1]))))

def absolute_distance():
    return Lambda(lambda tensors: K.abs(tensors[0]- tensors[1]))


class Distance(Enum):
    ABSOLUTE = absolute_distance
    COSINE = cosine_distance
    EUCLIDIAN = euclidian_distance


class Comparator(object):

    def __init__(self, input_size=128, hidden_layers = 2 , distance = Distance.ABSOLUTE, loss=0):

        # self.__input_reference = self.__input_layer('reference_layer')
        # self.__input_test = self.__input_layer('test_layer')
        # assert (isinstance(distance, Enum))
        self.__model = None
        self.__input_size = input_size
        self.__input_shape = (None, self.__input_size)
        self.__hidden_layers = hidden_layers
        self.__weight_decay = 0.0005
        self.__weight_init = "he_normal"
        print(distance)
        self.__comparator_layer = distance()

    def __input_layer(self, name='input'):
        return Input(shape=self.__input_shape, name=name)

    def __fully_connected_layer(self, layer, neurons=1024, name='dense', activation='relu'):
        return Dense(units=neurons, activation=activation, name=name, kernel_initializer=self.__weight_init,
                     kernel_regularizer=l2(self.__weight_decay))(layer)

    def get_model(self):
        if self.__model is None:
            self.__call__()
        return self.__model

    def __call__(self, *args, **kwargs):
        input_reference = self.__input_layer('reference_layer')
        input_test = self.__input_layer('test_layer')
        # ref_fc_layer1 = self.__fully_connected_layer(input_reference, neurons=256, name='ref_dense_layer1',
        #                                                   activation='relu')
        # test_fc_layer1 = self.__fully_connected_layer(input_test, neurons=256, name='test_dense_layer1',
        #                                                    activation='relu')
        # ref_fc_layer2 = self.__fully_connected_layer(ref_fc_layer1, neurons=128, name='ref_dense_layer2',
        #                                             activation='relu')
        # test_fc_layer2 = self.__fully_connected_layer(test_fc_layer1, neurons=128, name='test_dense_layer2',
        #                                              activation='relu')
        ref_layer = input_reference
        test_layer = input_test
        for l in range(self.__hidden_layers):
            ref_layer = self.__fully_connected_layer(ref_layer, neurons=(self.__hidden_layers - l) * self.__input_size , name='ref_dense_layer' + str(l+1),
                                                                                           activation='relu')
            test_layer = self.__fully_connected_layer(test_layer, neurons=(self.__hidden_layers - l) * self.__input_size,
                                         name='ref_test_layer' + str(l+1),
                                         activation='relu')
        # comparator_layer = Lambda(lambda tensors: K.abs(K.square(tensors[0]) - K.square(tensors[1])))
        vector_distance = self.__comparator_layer([ref_layer, test_layer])
        prediction = Dense(units=1, activation='sigmoid', name='output')(vector_distance)
        self.__model = Model(inputs=[input_reference, input_test], outputs=prediction)
        return self.get_model()

