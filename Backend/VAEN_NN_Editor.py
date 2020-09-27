import kerassurgeon.operations as operations
import tfkerassurgeon.operations as tfoperations


def delete_node(model, layer, node_indices):
    '''
    :param Model model: The tensorflow or keras model from which we wish to remove a node
    :param str or int layer: The name or index of the layer in which a node will be removed
    :param list node_indices: a list of the indices of the nodes that are to be removed
    '''

    assert isinstance(node_indices, list), "Please pass node_indices as a list of numbers"
    if isinstance(layer, int):
        try:
            return tfoperations.delete_channels(model, model.get_layer(index=layer), node_indices)
        except:
            try:
                return operations.delete_channels(model, model.get_layer(index=layer), node_indices)
            except:
                print("Please make sure you're satisfying the version requirements")
    if isinstance(layer, str):
        try:
            return tfoperations.delete_channels(model, model.get_layer(layer), node_indices)
        except:
            try:
                return operations.delete_channels(model, model.get_layer(layer), node_indices)
            except:
                print("Please make sure you're satisfying the version requirements")
