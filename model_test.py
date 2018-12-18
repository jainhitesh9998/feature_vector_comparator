from keras.layers import Input, Dense, concatenate, Lambda
from keras.models import Model
from keras import backend as K

from comparator.comparator import Comparator

model = Comparator( input_size=256)()
model.summary()