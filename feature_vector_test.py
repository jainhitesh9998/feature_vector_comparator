from random import randint

from feature_vector.feature_vector import FeatureVector

feature_vector = FeatureVector()
for i in range(100):
    feature_vector.add_vector(randint(0,10), [randint(0,100) for _ in range(10)])

output = feature_vector.get_vector_dict()

for keys, val in output.items():
    print(keys)
    print(len(val[0]))
    print()