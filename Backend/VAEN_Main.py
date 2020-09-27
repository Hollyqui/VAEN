import time
import threading
import thread_utils
import VAEN_NN_Editor
import VAEN_Data_Extractor
import numpy as np
import json

class VAEN():
    def __init__(self, update_frequency=1):
        self.running = True
        self.active = False
        self.models = {}
        self.update_frequency = update_frequency

    def interrupt(self):
        self.running = True
        self.active = False

    def stop(self):
        self.running = False
        self.active = False

    @thread_utils.actor(daemon=False)
    def add_model(self, model, name):
        self.models[name] = model

    def set_update_frequency(self, seconds):
        self.update_frequency = seconds

    def get_models(self):
        '''Returns a dictionary of model names and objects used by VAEN'''
        return self.models

    def get_models_json(self):
        '''Returns a json readable list of model names used by VAEN'''
        return json.dumps(list(self.models.keys()))

    @thread_utils.actor(daemon=False)
    def get_structure(self, model):
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

    @thread_utils.actor(daemon=False)
    def get_weights_json(self, model_name, layer_index):
        '''Returns a json formatted string containing the weights of the specified
        layer and model'''
        return json.dumps({"weights": self.models[model_name].get_weights()[layer_index].tolist()})


    @thread_utils.actor(daemon=False)
    def get_all_weights(self, model_name):
        '''Returns a numpy array of all weights from a specified model'''
        return self.models[model_name].get_weights()

    @thread_utils.actor(daemon=True)
    def updater(self):
        while self.active==True:
            for model_name in self.models:
                # print("Model information:")
                # print(self.get_structure(self.models[model_name]).receive())
                time.sleep(self.update_frequency)



    @thread_utils.actor(daemon=True)
    def start(self):
        self.running = True
        self.active = True
        print("test")
        self.updater()

#
# vaen = VAEN()
# vaen.start()
# vaen.stop()
# vaen.active
# vaen.running
# vaen.set_update_frequency(0.1)
#
#
# 2+2
