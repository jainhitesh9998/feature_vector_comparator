from keras.layers import Input, Dense, concatenate, Lambda
from keras.models import Model
from keras import backend as K
from comparator.comparator import Distance
from comparator.comparator import Comparator
from keras.applications.resnet50 import ResNet50
# reference_resnet = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# test_resnet = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# test_resnet.summary()

# print(type(test_resnet.layers[-2][0]))
# print(test_resnet.layers[-3].shape)
# model = Comparator(input_size=test_resnet.layers[-2].output.shape[1], hidden_layers=2, distance=Distance.ABSOLUTE, \
#     reference_Model= reference_resnet, test_model=None)()

model = Comparator(input_size=128, hidden_layers=2, distance=Distance.EUCLIDIAN)()
model.summary()