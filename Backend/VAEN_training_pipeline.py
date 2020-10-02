import pickle
import tensorflow as tf
import time
from functools import wraps
import os
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import thread_utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

def get_sub_dirs(path):
    return [d for d in os.listdir(str(path)) if os.path.isdir(os.path.join(str(path)+os.sep, d))]


@thread_utils.actor(daemon=True)
def read_images(path):
    '''This method reads in a folder of images with equal size'''
    data = []
    labels = []
    # iterate through all sub directories

    sub_dirs = get_sub_dirs(path)
    sub_path = Path.home().joinpath(path,sub_dirs[0])
    extensions = ['.jpg', '.png', '.jpeg']
    file_path = [x for x in Path(sub_path).iterdir() if x.suffix.lower() in extensions][0]
    print("Image shape set to:",cv2.imread(str(file_path)).shape)
    h = cv2.imread(str(file_path)).shape[0]
    w = cv2.imread(str(file_path)).shape[1]
    c = cv2.imread(str(file_path)).shape[2]

    for label,subdir in enumerate(sub_dirs):
        sub_path = Path.home().joinpath(path,subdir)
        file_paths = [x for x in Path(sub_path).iterdir() if x.suffix.lower() in extensions]
        #read dimension
        data_points = len(file_paths)

        print(len([x for x in Path(sub_path).iterdir()])-data_points, "file(s)",
              "out of",len([x for x in Path(sub_path).iterdir()])," in the directory",
              "are not a .jpg, .png or .jpeg file and were therefore not read")




        # load images
        imgs = np.empty((data_points,h,w,c), dtype='uint16')
        for num,img in enumerate(file_paths):
            try:
                imgs[num,:,:,:] = cv2.imread(str(img))
            except:
                print("Image",str(img),"seems to have wrong dimensions",cv2.imread(str(img)).shape,"instead of",(h,w,c),"and wasn't read!")
            # Storing along the last two axes
        temp_labels = np.ones((data_points))*label
        data += imgs.tolist()
        labels += temp_labels.tolist()
        output_shape = (len(sub_dirs),)

    data = np.array(data, ndmin=4)
    data = data/np.max(data)
    return data, np.array(labels, ndmin=1), output_shape



def dataset_split(x,y,train_size=0.8,random_state=None, shuffle=True):
    """Returns X_train, X_test, y_train, y_test in this order"""
    return train_test_split(x,y,train_size=train_size,random_state=random_state,shuffle=shuffle)


def to_grey_scale(data):
    """Converts RGB & RGBA images to greyscale"""
    r, g, b = data[:,:,:,0], data[:,:,:,1], data[:,:,:,2]
    return np.reshape(np.array(0.2989 * r + 0.5870 * g + 0.1140 * b,ndmin=3),
                      (data.shape[0],data.shape[1],data.shape[2],1))


# def create_network(input_shape, n_labels):
#     return  tf.keras.models.Sequential([
#             tf.keras.layers.Flatten(input_shape=input_shape),
#             tf.keras.layers.Dense(128, activation='relu'),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(n_labels)
#             ])
#
#
#
#
# def train(model, x, y):
#     model.fit(x, y, epochs=5)


#
#
# path = Path("C:/Users/felix/Documents/AtomProjects/VAEN/data/MNIST/trainingSet/trainingSet")
#
# x, y, n_classes = read_images(path).receive()
#
# n_classes
# x_grey = to_grey_scale(x)
# model = compile(create_network((28,28,1), 10))
# train(model, x_grey, y)
