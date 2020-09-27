import tensorflow as tf
import thread_utils
from VAEN_Main import VAEN
from threading import Thread
import time
import random


def VAEN_thread(func, *args, **kwargs):
    thread = Thread(target=func, args=args, kwargs=kwargs)
    thread.start()

    # thread2 = Thread(target=count_me_rev(5))
    # thread2.start()

    return thread


class VAEN_Callback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.training = True
        self.backup_model = None
        self.batch = 0
        self.epoch = 0
        self.loss = []
        self.id = str(int(time.time()))+str(random.randint(100000,999999))

    def get_id(self):
        '''Return unique identifier ID'''
        return self.id

    def on_epoch_end(self, epoch, logs={}):
        if not self.training:
            print("Training has been stopped manually")
            self.model.stop_training = True
        # print("Current loss is", logs["loss"])


    def get_loss(self):
        return self.loss[-1][-1]

    def on_train_batch_end(self, batch, logs={}):
        # self.loss.append((self.epoch, self.batch, logs["loss"]))
        self.epoch += 1

        self.batch += 1


    def stop_training(self):
        self.training = False
        print("Training will be stopped after the next epoch")
        # self.model.stop_training = True

    def resume_training(self):
        self.training = True
        print("Training will be stopped after the next epoch")
        self.model.stop_training = False


#
#
# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])
#
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#
# model.compile(optimizer='adam',
#               loss=loss_fn,
#               metrics=['accuracy'])
#
# vaen = VAEN(update_frequency=3)
# vaen.set_update_frequency(5)
# vaen.add_model(model, "dense_net")
# vaen.start()
#
# cb = VAEN_Callback()
#
# VAEN_thread(model.fit, x_train, y_train, epochs=15, callbacks=[cb])
#
# cb.get_loss()
#
#
# vaen.stop()
# cb.stop_training()
# cb.resumte_training()
#
#
# import numpy as np
# vaen.get_all_weights("dense_net").receive()[0].shape
#
# vaen.get_structure(model).receive()
# vaen.stop()
