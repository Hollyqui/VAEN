from VAEN_Main import VAEN
import VAEN_NN_Editor
from VAEN_callbacks import VAEN_Callback
from VAEN_callbacks import VAEN_thread
import VAEN_training_pipeline
import tensorflow as tf
import traceback
import numpy as np
import thread_utils


class VAEN_Auto_ML:

    def __init__(self):
        self.vaen = None
        self.cb = None
        self.model = None
        self.compiler = None
        self.loss = None
        self.output_shape = None
        self.input_shape = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None


    def create_default_model(self, input, n_classes):
        """Creates a standard network that can then be expanded/reduced
        using editor functions"""
        input_shape = input.shape[1:]
        return  tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(int(n_classes))
                ])

    def import_custom_loss(self, custom_loss_function, output_shape):
        """Imports the custom loss function of a classifier net and ensures
        that it can compute the correct output shape. The custom function MUST
        be called 'loss()', else the program won't run"""
        output_shape = str(output_shape)
        function_call = custom_loss_function+"\n"+"np.sum(loss(np.zeros("+output_shape+"),0))"
        # result = np.ndarray((output_shape))
        try:
            exec(function_call, globals())
            self.loss = loss
        except Exception as e:
            print('Failed to use custom loss function:')
            traceback.print_exc()



    def import_custom_model(self, code=None, path_to_model=None):
        """Imports a custom tensorflow model either based on user code
        or based on already existing model that is loaded via tf2.0. The custom
        code MUST contain a method called create_custom_network() that returns
        the net"""
        model = None
        if code is not None:
            if path_to_model is not None:
                print("code & model passed - only the uploaded model will be",
                      "used, if you wish to use the code please delete the uploaded",
                      "model first")
            try:
                exec(code, globals())
                model = create_custom_network()
                #just check that the code actually creates a model and has a summary function
                model.summary()
            except Exception as e:
                print('Failed to use custom model:')
                traceback.print_exc()
        if path_to_model is not None:
            try:
                model = tf.keras.models.load_model(path_to_model)
                model.summary()
            except Exception as e:
                print('Failed to use custom model:')
                traceback.print_exc()
        self.model = model
        return model

    patha = "C:/Users/felix/Documents/AtomProjects/VAEN/data/saved_model.h5"

    # @thread_utils.actor(daemon=True)
    def custom_compile(self):
        # TODO: Implement
        pass


    # @thread_utils.actor(daemon=True)
    def default_compile(self, model):
        if self.loss is None:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss_fn = self.loss
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        return model

    # @thread_utils.actor(daemon=True)
    def init_network(self):
        """This function checks creates & compiles the network in order to then
        do automl on it"""
        # network is created and compiled using either custom or
        # default functions (if no custom functions are loaded)
        if self.model is None:
            starting_model = self.create_default_model(self.x_train, self.output_shape[0])
            print("Default Starting Model Loaded")
        else:
            starting_model = self.model
            print("Custom Model Loaded")

        self.cb = VAEN_Callback()

        if self.compiler is None:
            compiled_model = self.default_compile(starting_model)
            print("Default Compiler Loaded", compiled_model.summary())
        else:
            compiled_model = self.custom_compile(starting_model)
            print("Custom Compiler Loaded", compiled_model.summary())
        print
        self.vaen = VAEN(update_frequency=1)
        self.vaen.add_model(compiled_model, "starting_model")

    def train_network(self, epochs=20):
        self.vaen.get_model('starting_model').fit(self.x_train, self.y_train, epochs=epochs,callbacks=self.cb)

    def load_data(self, path):
        x, y, self.output_shape = VAEN_training_pipeline.read_images(path).receive()
        self.x_train, self.x_test, self.y_train, self.y_test = VAEN_training_pipeline.dataset_split(x, y)

    def full_auto(self, path, epochs=20):
        self.load_data(path)
        self.init_network()
        self.train_network(epochs=epochs)

from pathlib import Path
from AutoML import VAEN_Auto_ML
#
# path = Path("C:/Users/felix/Documents/AtomProjects/VAEN/data/MNIST/trainingSet/trainingSet")
# auto_ml = VAEN_Auto_ML()
#
# auto_ml.full_auto(path)

#
#
# from pathlib import Path
#
#
# path = Path("C:/Users/felix/Documents/AtomProjects/VAEN/data/MNIST/trainingSet/trainingSet")
# auto_ml = VAEN_Auto_ML()
#
# x, y, output_shape = VAEN_training_pipeline.read_images(path).receive()
# X_train, X_test, y_train, y_test = VAEN_training_pipeline.dataset_split(x, y)
# auto_ml.init_network(X_train, X_test, y_train, y_test, output_shape)
# auto_ml.vaen.get_model('starting_model').fit(X_train, y_train, epochs=5,callbacks=auto_ml.cb)
#
#
#
#
#
# testvaen = VAEN(update_frequency=1)
# testvaen.get_models_json()
#
#
# x_grey = VAEN_training_pipeline.to_grey_scale(x)
#
# custom_loss_function = """
# import numpy as np
# def loss(output, label):
#     return((output-label)**2)
# """
# import_classification_loss(custom_loss_function, output_shape)
#
# input = """
# import tensorflow as tf
# def create_custom_network():
#     model = tf.keras.models.Sequential([
#             tf.keras.layers.Flatten(input_shape=(28,28,1)),
#             tf.keras.layers.Dense(128, activation='relu'),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(10, activation='softmax')
#             ])
#     return model
#     """
# model = import_custom_model(input)
#
#
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(optimizer='adam',
#               loss=loss,
#               metrics=['accuracy'])
# x.shape
# model.fit(x_grey/255.0, y, epochs=5, shuffle=True)
#
# import matplotlib.pyplot as plt
# import random
#
# rand = random.randint(0,42000)
# plt.imshow(x[rand])
# np.argmax(model.predict(x_grey[rand-1:rand]))
