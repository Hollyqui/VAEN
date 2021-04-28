# TODO:
# - add position id (sometimes the same layer might appear multiple times in one network)
# - optimize the node & layer removal methods
# - clean up code


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import graphviz
import os
from collections import defaultdict
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'
import datetime
# tf.compat.v1.disable_eager_execution()
import json
import time
import random

class Layer:
    def __init__(self, tf_layer, display_name, type, unique_id=None, position_id = None):
        """tf_layer should be a tf.keras layer, id needs to be a unique id
        and type is either 'Input, Layer or Output'."""
        self.tf_layer = tf_layer
        if unique_id is None:
            self.unique_id = str(tf_layer)[-20:-1].replace("x","y").replace(" ","X")
        else:
            self.unique_id = unique_id
        if position_id is None:
            self.position_id = str(time.time()+random.random())
        else:
            self.position_id = position_id
        # self.tf_layer._name = self.position_id
        self.display_name = display_name
        self.tf_layer._name = self.unique_id
        self.inp = {}
        self.out = {}


        try:
            self.weights = tf_layer.get_weights()
        except:
            self.weights = None
        self.type = type

    def get_layer_info(self):
        """Return Layer info"""
        info = []
        if self.type=="Input":
            info.append({})
            info[-1]["shape"] = list(self.tf_layer.get_shape())
            info.append({})
        else:
            info.append(layers.serialize(self.tf_layer))
            try:
                # if layer is initialized and has weights
                info[-1]["avg_weight"] = str(np.average(self.tf_layer.get_weights()[0]))
                info[-1]["avg_bias"] = str(np.average(self.tf_layer.get_weights()[1]))
                info[-1]["avg_abs_weight"] = str(np.average(np.abs(self.tf_layer.get_weights()[0])))
                info[-1]["avg_abs_bias"] = str(np.average(np.abs(self.tf_layer.get_weights()[1])))
            except:
                # if layer is not initialized it won't have weights
                info[-1]["avg_weight"] = 0
                info[-1]["avg_bias"] = 0
                info[-1]["avg_abs_weight"] = 0
                info[-1]["avg_abs_bias"] = 0
            info.append({})
        info[-1]["id"] = self.position_id
        info[-1]["type"] = self.type
        info[-1]["Inputs"] = json.dumps(list(self.inp.keys()))
        info[-1]["Outputs"] = json.dumps(list(self.out.keys()))
        return info

    def save_weights(self, weights=None):
        """Saves weights of a tf_layer to the Layer object"""
        if weights is None:
            self.weights = self.tf_layer.get_weights()
        else:
            self.weights = weights

    def set_weights(self, weights=None):
        """Sets the weight of the tf_layer to what is saved in the Layer object"""
        if weights is None:
            weights = self.weights
        self.tf_layer.set_weights(weights)

    def set_input(self,layer):
        """either pass a layer as input"""
        self.inp[layer.position_id] = layer

    def set_output(self,layer):
        self.out[layer.position_id] = layer

    def remove_input(self,position_id):
        self.inp.pop(position_id)

    def remove_output(self,position_id):
        self.out.pop(position_id)

    def change_nodes(self, new_nodes):
        # rebuild layer with 5 less nodes
        # new_nodes = len(self.weights[1])-number_of_nodes

        serialized = layers.serialize(self.tf_layer)
        serialized["config"]["units"] = new_nodes
        self.tf_layer = layers.deserialize(serialized)
        # load back old weights
        self.save_weights([self.weights[0][:,:new_nodes],
                   self.weights[1][:new_nodes]])

class Network_Constructor:
    def __init__(self, network_id):
        # self.graph = graph = tf.Graph.__init__()
        self.position_id = network_id

        self.inputs = []
        self.layers = {} # layers indexed by their position id
        self.unique_layers = defaultdict(list) # layers indexed by their unique id
        self.outputs = []

        self.connected_layers = {}
        self.connected_inputs = []
        self.connected_outputs = []

        self.connections_out = defaultdict(list)
        self.connections_in = defaultdict(list)
        self.connections = defaultdict(list)

        self.compiled_model = None

    def get_network_info(self):
        return json.dumps([layer.get_layer_info() for layer in list(self.layers.values())])

    def add_layers(self, layers):
        """Adds a Layer to the Network"""
        tf.keras.backend.clear_session()
        if isinstance(layers,list)==False:
            layers = [layers]
        for layer in layers:
            self.layers[layer.position_id] = layer
            if layer.type == "Input":
                self.inputs.append(layer.position_id)
            if layer.type == "Output":
                self.outputs.append(layer)
    def create_unique_dict(self):
        for layer in self.layers.values():
            self.unique_layers[layer.unique_id].append(layer)

    def remove_input(self, id1, id2):
        """Removes a connection between layers"""
        self.layers[id2].inp.pop(id1)
        self.layers[id1].out.pop(id2)

    def set_input(self, id1,id2):
        """Adds a connection between layers"""
        self.layers[id2].set_input(self.layers[id1])
        self.layers[id1].set_output(self.layers[id2])

    def remove_layer(self, position_id):
        """Please try never to use this until suuuper necessary for some reason,
        it's the epitome of inefficient"""
        self.layers.pop(position_id)
        for layer in self.layers.values():
            if position_id in layer.inp:
                layer.inp.pop(position_id)
            elif position_id in layer.out.values():
                layer.out.pop(position_id)



    def get_layer(self, position_id=None, unique_id=None):
        """Finds a layer by position_id/unique_id from a tensorflow model"""
        try:
            return [layer for layer in self.compiled_model.layers if layer.position_id == position_id][0]
        except:
            return [layer for layer in self.compiled_model.layers if layer.position_id == unique_id][0]
        finally:
            return None

    def find_value(self, item,dict):
        """Finds value in a dictionary and returns all the occurances. Second return values
        returns True if at least one instance is found"""
        occurances = []
        for tuple in list(dict.items()):
            if item in tuple[1]:
                occurances.append(tuple)
        return occurances, len(occurances)>0


    def save_weights(self):
        # iterate through layer ids
        for layer in list(self.layers.values()):
            tf_lay = self.get_layer(layer.name, self.compiled_model)
            if tf_lay is not None:
                layer.save_weights(tf_lay.get_weights())

    def connect_layers(self):
        """"""
        self.connections_in = defaultdict(list)
        self.connections_out = defaultdict(list)

        for layer in self.layers.values():
            if len(layer.inp)>1:
                for key in list(layer.inp.keys()):
                    self.connections_in[layer.position_id].append(key)

        for layer in self.layers.values():
            if len(layer.out)>=1:
                for key in list(layer.out.keys()):
                    # print(key)
                    if key not in list(self.connections_in.keys()):
                        # print("key not in dict_in", key)
                        self.connections_out[layer.position_id].append(key)


    def parse_model(self, model):
        """Reads in a tensorflow model"""
        # first clear all variables
        self.layers = {}
        self.connected_layers = {}
        self.connected_inputs = []
        self.connected_outputs = []
        self.inputs = []
        self.outputs = []

        # ensure all layer names are unique
        unique_ids = []
        position_ids = []
        vaen_layers = {}
        for idx, layer in enumerate(model.layers):
            print(layer.name)
            if layer.name in position_ids:
                layer._name = layer.name+str(idx)
            position_ids.append(layer._name)
            # if str(layer) not in unique_ids:
                # print(layer.name)
            vaen_layers[layer.name] = Layer(layer, unique_id=layer.name, position_id=layer.name, display_name = layer.name, type = "Layer")
            unique_ids.append(layer.name)

        # iterate through all layers of the tensorflow model
        for layer in model.layers:
            # disconnected_layer = layers.deserialize({'class_name': layer.__class__.__name__, 'config': layer.get_config()})
            disconnected_layer = layers.deserialize(layers.serialize(layer))
            weights = layer.get_weights()
            # add the layer to the self.layers array
            layer_to_add = vaen_layers[layer.name]
            self.layers[vaen_layers[layer.name].position_id] = layer_to_add

            # save the weights to the layer (NOTE this doesn't mean they're loaded to
            # the tensorflow layer, they're only saved to the custom layer object)
            self.layers[layer.name].weights = weights

            # if there is only a single or no input make it into a list of length 1
            # so it's iterable
            if isinstance(layer._inbound_nodes[0].inbound_layers,list):
                input_layer = layer._inbound_nodes[0].inbound_layers
            else:
                input_layer = [layer._inbound_nodes[0].inbound_layers]

            # if the input_layer is empty we know that the layer has to be an
            # input
            if len(input_layer)<1:
                self.layers[layer_to_add.position_id] = Layer(layer.input, position_id=layer.name, display_name = layer.name, type = "Input")
                self.inputs.append(layer.name)
            # if layer.position_id not in self.layers:
            # Otherwise we iterate though the inputs and set them accordingly
            for input in input_layer:
                self.set_input(input.name,layer.name)
                # print(layer.position_id,"gets input from",input.position_id)
            self.create_unique_dict()

        self.connect_layers()
        for layer in self.layers:
            if layer not in list(self.connections_out.keys()):
                _, in_conv = self.find_value(layer,self.connections_in)
                if not in_conv:
                    self.layers[layer].type = "Output"
                    self.outputs.append(self.layers[layer])
    # computer_vision_net.connect_layers()
    # parse_model(computer_vision_net,mobile)
    # for l in computer_vision_net.layers.values():
    #     print(l.type)


    def parse_connections(self):
        """Creates an array containing all layer(i+1) --> layer(i) connections"""
        self.connections = defaultdict(list)
        for layer in self.layers.keys():
            connects_from = list(self.layers[layer].inp.keys())
            if len(connects_from) > 0:
                self.connections[layer] = connects_from
        return self.connections

    def get_connections_of_output(self,out):
        """Creates a nested dictionary that stores all connections - this is
        later used to find out in which order the connections have to be called"""
        # creates dictionary to store the correct order
        order = defaultdict(list)
        # if the output layer has a connection (everything up to input)
        if out in list(self.connections.keys()):
            # for all connections that layer has
            for layer in self.connections[out]:
                print(layer, out)
                order[out].append(layer)
                temp = self.get_connections_of_output(layer)
                if temp is not None:
                    # print(layer, temp[layer])
                    order[layer].append(temp)
            return order

    def get_all_connections(self):
        """Gets a nested dictionary of all connections - will be used to reconstruct
        the correct connection order"""
        ordered_list = []
        for output in self.outputs:
            print(output, self.get_connections_of_output(output.position_id))
            ordered_list.append(self.get_connections_of_output(output.position_id))
        return ordered_list

    def get_order(self, start, nested, connections_reverse):
        """Finds in which order connections need to be established to create the
        tensorflow model - this only looks at a single output"""
        if isinstance(nested,defaultdict):
            all_inputs = list(nested.keys())
            if len(all_inputs)==1:
                if [nested[all_inputs[0]],start] not in connections_reverse:
                    connections_reverse.append([nested[all_inputs[0]],start])
            inputs = [input for input in all_inputs if input is not start]
            all_inputs = [input for input in all_inputs]

            for input in all_inputs:
                if input is not start:
                    if [inputs,start] not in connections_reverse:
                        connections_reverse.append([inputs,start])
                self.get_order(input,nested[input][0],connections_reverse)
        return connections_reverse

    def get_connections_in_order(self):
        """Gets all (from all outputs) connections needed to create the tensorflow
        model in correct order"""
        self.parse_connections()
        ordered_list = self.get_all_connections()
        connections = []
        connections_inverse = []
        connections_output = []
        for idx, output in enumerate(self.outputs):
            connections_output.append([])
            connection = self.get_order(output.position_id,ordered_list[idx],connections_inverse)
            for con in connection:
                if con not in connections:
                    connections.append(con)
                    connections_output[-1].append(con)
            connections_output[-1].reverse()
        return connections_output

    def reset_layers(self):
        for layer in self.layers:
            if self.layers[layer].type != "Input":
                self.layers[layer].tf_layer = layers.deserialize(layers.serialize(self.layers[layer].tf_layer))
            else:
                self.layers[layer].tf_layer._name = self.layers[layer].position_id

    def set_weights(self):
        for layer in self.compiled_model.layers:
            try:
                layer.set_weights(self.layers[layer.name].weights)
            except:
                print("Weight loading not possible for layer",layer.name)

    def save_weights(self):
        for layer in self.compiled_model.layers:
            for layer_instance in self.unique_layers[layer.name]:
                layer_instance.save_weights(layer.get_weights())

    def change_nodes(self, layer_name, number_of_nodes):
        self.layers[layer_name].change_nodes(number_of_nodes)
        self.compile()

    def compile(self, load_weights=True):
        """Constructs the tensorflow model"""
        self.connected_layers = {}
        self.connected_inputs = []
        self.connected_outputs = []
        self.reset_layers()

        cons = self.get_connections_in_order()
        for outputs in cons:
            for connections in outputs:
                output = connections[1]

                for input in connections[0]:
                    if self.layers[input].type == "Input":
                        self.connected_layers[input] = self.layers[input].tf_layer
                        self.connected_inputs.append(self.connected_layers[input])
                # check if it's a concatenation
                if len(connections[0])>1:
                    input_layers = [self.connected_layers[input] for input in connections[0]]
                    self.connected_layers[output] = self.layers[output].tf_layer(input_layers)

                else:
                    input = connections[0][0]
                    # self.connected_layers[input]._name = self.layers[input].position_id
                    # print(self.connected_layers[input].position_id,self.layers[input].position_id,input)
                    # print(self.layers[output].tf_layer, self.connected_layers[input])
                    self.connected_layers[output] = self.layers[output].tf_layer(self.connected_layers[input])
                if self.layers[output].type == "Output":
                    self.connected_outputs.append(self.connected_layers[output])
        #### # DEBUG:
        # print(self.connected_inputs[0].position_id)
        # [print(l.name) for l in self.connected_layers.values()]
        self.compiled_model = keras.Model(
            inputs=self.connected_inputs,
            outputs=self.connected_outputs)
        self.create_unique_dict()
        if load_weights:
            self.set_weights()
        self.save_weights()
        return self.compiled_model

# TODO:
# - instead of layerid use positionid & layerid


# resnet = tf.keras.applications.ResNet50(
#     include_top=True, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None, classes=1000)
# mobile = tf.keras.applications.MobileNetV2(
#     input_shape=None, alpha=1.0, include_top=True, weights='imagenet',
#     input_tensor=None, pooling=None, classes=1000,
#     classifier_activation='softmax')
# computer_vision_net = Network_Constructor("cv")
# start = time.time()
# computer_vision_net.parse_model(mobile)
# end = time.time()
# print(end-start)
# # mobile.summary()
# # computer_vision_net.layers
#
# # plot_model(mobile, "multi_input_and_output_model.png", show_shapes=True)
#
#
# computer_vision_net.reset_layers()
# computer_vision_net.parse_connections()
# computer_vision_net.get_connections_of_output("predictions")



# computer_vision_net.get_connections_of_output()


# computer_vision_net.get_all_connections()


# start = time.time
# computer_vision_net.get_connections_in_order()


# reconstructed_net = computer_vision_net.compile()
# computer_vision_net.unique_layers['X0y0000017AD1057F10>'][1].position_id
#
#
# end = time.time()
# computer_vision_net.connected_layers["block_14_add"]
# computer_vision_net.connected_inputs
# computer_vision_net.connected_layers
# computer_vision_net.get_connections_in_order()
#
# reconstructed_net.summary()
# computer_vision_net.layers
# computer_vision_net.layers["input_1"].type
#


#############################
layers.Dense(units=69)

network = Network_Constructor("Net3209")
# create all layers
inpu = Layer(layers.Input(shape=(160,),name="Input0"),display_name = "sdkfjsdkfa", position_id =  "Input0", type="Input")
inpu2 = Layer(layers.Input(shape=(142,123), name="Input1"), display_name = "sdkfjsdkfa",position_id = "Input1", type="Input")
inpu3 = Layer(layers.Input(shape=(34,), name="Input2"),display_name = "sdkfjsdkfa",position_id =  "Input2", type="Input")
a = Layer(layers.Dense(units=69),display_name = "sdkfjsdkfa", position_id = "Layer1", type="Layer")
b = Layer(layers.Dense(units=230),display_name = "sdkfjsdkfa",position_id =  "Layer2", type="Layer")
c = Layer(layers.LSTM(units=85),display_name = "sdkfjsdkfa",position_id = "Layer3", type="Layer")
d = Layer(layers.Flatten(),display_name = "sdkfjsdkfa",position_id =  "Layer4", type="Layer")
e = Layer(layers.Dense(units=72, name="Test_layer5"),display_name = "sdkfjsdkfa",position_id =  "Layer5", type="Layer")
# e.tf_layer.input
f = Layer(layers.Dense(units=159, name="Test_layer6"),display_name = "sdkfjsdkfa",position_id =   "Layer6", type="Layer")
output1 = Layer(layers.Dense(units=39, name="Test_Output"), display_name = "sdkfjsdkfa",position_id =  "Output3", type = "Output")
output2 = Layer(layers.Dense(units=100, name="Test_Output2"),display_name = "sdkfjsdkfa",position_id =   "Output4", type = "Output")
conc = Layer(layers.Concatenate(axis=-1, name='Conc5'),display_name = "sdkfjsdkfa",position_id =   "Concatenate5", type="Layer")
conc2 = Layer(layers.Concatenate(axis=-1, name='Conc6'),display_name = "sdkfjsdkfa",position_id =   "Concatenate6", type="Layer")

len(a.unique_id)
# add them to the network
network.add_layers(inpu)
network.add_layers([a,b,c,d,e,f,conc,conc2,output1,output2,inpu2,inpu3])
# now specify inputs
network.set_input("Input0","Layer2")
network.set_input("Input1","Layer3")
network.set_input("Input2","Layer1")
network.set_input("Layer3","Layer4")
network.set_input("Layer1","Concatenate5")
network.set_input("Layer4","Concatenate5")
network.set_input("Layer2","Concatenate5")
network.set_input("Concatenate5","Concatenate6")
network.set_input("Layer2","Concatenate6")
network.set_input("Concatenate6","Layer5")
network.set_input("Layer5","Layer6")
network.set_input("Layer6","Output3")
network.set_input("Concatenate5","Output4")


# c.tf_layer(inpu2.tf_layer)
# construct the network
import time
start = time.time()
model = network.compile()
network.layers
model.layers
network.get_all_connections()
end = time.time()
print(end-start)

# now check that everything is correct
model.summary()
plot_model(model, "multi_input_and_output_model_1.png", show_shapes=True)


model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
# create another network constructor
parsed_network = Network_Constructor("Net3399")
# now read in a model (feel free to load any other model that you can
# find online)
start = time.time()
parsed_network.parse_model(model)
end = time.time()
print(end-start)
# now reconstruct the model
start = time.time()
parsed_model = parsed_network.compile()
end = time.time()
print(end-start)
parsed_network.set_weights()




# check if everything is correct
# parsed_model.summary()
# plot_model(parsed_model, "multi_input_and_output_model.png", show_shapes=True)
# now let's remove some a layer from the newly constructed model
# first disconnect it (I'll make this process easier later)
# then remove the layer
# parsed_network.remove_layer("Layer5")
# parsed_network.layers["Input1"].out
# and connect everything back
# parsed_network.set_input("Concatenate6", "Layer6")
# now let's remove another connection without removing a layer
# parsed_network.remove_input("Layer2","Concatenate5")




#  check if everything is correct
#  parsed_network.layers["Layer6"].change_nodes(100)
#  parsed_model = parsed_network.compile()
#  parsed_model.summary()
#  plot_model(parsed_model, "multi_input_and_output_model_1.png", show_shapes=True)


#
#
#
mnist_net = Network_Constructor("932053")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
mnist_layers = [
    # THIS
    # Layer(layers.Input(shape=(28*28,),name="Input1"),position_id="Input1",display_name="Input", type="Input"),
    Layer(layers.Input(shape=(28,28),name="Input1"),position_id="Input1",display_name="Input", type="Input"),
    Layer(tf.keras.layers.Flatten(input_shape=(28, 28)),position_id="Flatten2",display_name="Flatten",type="Layer"),
    Layer(tf.keras.layers.Dense(128, activation='relu'),position_id="Dense3",display_name="Dense",type="Layer"),
    Layer(tf.keras.layers.Dropout(0.2),position_id="Dense4",display_name="Dense",type="Layer"),
    Layer(tf.keras.layers.Dense(10),position_id="Output5",display_name="Out",type="Output"),
]
mnist_net.add_layers(mnist_layers)
mnist_net.layers
mnist_net.layers["Input1"].tf_layer
mnist_net.set_input("Input1","Flatten2")
mnist_net.set_input("Flatten2","Dense3")
mnist_net.set_input("Dense3","Dense4")
mnist_net.set_input("Dense4","Output5")
mnist_net.layers["Input1"].position_id
mnist_net.connected_inputs
net = mnist_net.compile()
layer, node_index, tensor_index = net.outputs[0]._keras_history
layer
node_index
tensor_index
net.outputs

logs_base_dir = "./logs"
os.makedirs(logs_base_dir, exist_ok=True)
# !load_ext tensorboard
logdir = os.path.join(logs_base_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
net.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
net.fit(x_train, y_train, callbacks=[tensorboard_callback],epochs=5)
net.evaluate(x_test,  y_test, verbose=2)
net.summary()
plot_model(net, "multi_input_and_output_model_2.png", show_shapes=True)
