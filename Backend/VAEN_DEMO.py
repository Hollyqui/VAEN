from pathlib import Path
from AutoML import VAEN_Auto_ML



"""This is the absolute shortest way to train a classification dataset of images
by only using default methods."""

path = Path("C:/Users/felix/Documents/AtomProjects/VAEN/data/MNIST/trainingSet/trainingSet")
auto_ml = VAEN_Auto_ML()
auto_ml.full_auto(path)

#%%

 """If you want to inject your own methods that's also quite easy, for example if
 you would like to define a custom starting structure via code:"""

path = Path("C:/Users/felix/Documents/AtomProjects/VAEN/data/MNIST/trainingSet/trainingSet")
auto_ml = VAEN_Auto_ML()

# note that you have to create a function called create_custom_network
# that returns a tensorflow model (doesn't even need to be keras)
custom_structure = """
import tensorflow as tf
def create_custom_network():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28,3)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
            ])
    return model
    """

auto_ml.import_custom_model(code=custom_structure)
auto_ml.full_auto(path)

#%%

"""In a very similar fashion, you can also load a model from an existing model
file by just replacing the 'auto_ml.import_custom_model' line by:

auto_ml.import_custom_model(path_to_model="C\User\...\model.h5")

"""

#%%

"""In the same way, you are able to load a custom compiler or loss. By loading
the dataset, the auto_ml suit will automatically adjust to whatever is passed
if no custom code is uploaded"""
