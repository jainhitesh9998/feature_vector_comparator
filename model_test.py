from keras.layers import Input, Dense, concatenate, Lambda, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from comparator.comparator import Distance
from comparator.comparator import Comparator
from keras.applications.resnet50 import ResNet50
reference_resnet = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
test_resnet = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
test_resnet.summary()
reference_resnet.summary()
for layer in reference_resnet.layers:
    layer.name = layer.name + '_1'

print(type(test_resnet.layers[-2].output))
# print(test_resnet.layers[-3].shape)
# model = Comparator(input_size=test_resnet.layers[-2].output.shape[1], hidden_layers=2, distance=Distance.ABSOLUTE, \
#     reference_Model= reference_resnet, test_model=None)()

model = Comparator(input_size=2048, hidden_layers=2, distance=Distance.EUCLIDIAN, ref_model=reference_resnet, test_model=test_resnet)()
model.summary()
# print(model.layers[0].input)
# print(test_resnet.layers[-2].output)
# f1 = Flatten(name="flat_ref")(Reshape(target_shape=(None, None, test_resnet.layers[-2].output.shape[1]))(test_resnet.layers[-2].output))
# ref_in = Dense(name="ref_input", units=4096, activation='relu')(f1)
# test_in = Dense(name="test_input", units=4096, activation='relu')(Flatten(name="flat_test")(test_resnet.layers[-2]))
model.save("weights.h5")