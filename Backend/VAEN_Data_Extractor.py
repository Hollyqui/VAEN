import tfkerassurgeon
import json
import numpy as np


i = 0
# this loop gets the information about a neural net and makes it into a json
def get_structure(model):
    ''' Returns information about the structure of a network for visualisation
    purposes.'''
    network = []
    for i , layer in enumerate(model.layers):
        network.append(layer.get_config())
        network[-1]["id"] = i
        network[-1]["type"] = layer.__class__.__name__
        network[-1]["avg_weight"] = str(np.average(model.get_weights()[i]))
        network[-1]["avg_abs_weight"] = str(np.average(abs(model.get_weights()[i])))
    return json.dumps({"network": network})

def get_weights(model, layer_index):
    '''Returns the weights between the specified layer and the following layer'''
    return model.get_weights(layer)

# ########### Testing ###############

# print(json.dumps({"network": network}))
# print(get_structure(model))
#
#
# model.get_weights()[2].shape


#
# import tensorflow as tf
# from tensorflow import keras
# import numpy as np
# import pandas as pd
# import seaborn
# import json
# import codecs, json
# import io
#
#
#
#
#
# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# model = keras.Sequential([
#     keras.layers.Dense(40, input_shape=(28,28,),activation='relu'),
#     keras.layers.Dense(20, activation='relu'),
#     keras.layers.Dense(10, activation='relu'),
#     keras.layers.Dropout(rate=0.1),
#     keras.layers.Dense(3)
#     ])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
#
# print(get_structure(model))
