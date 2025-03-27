import sys
import os 
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)


from data_manager import data_loader
from model_creator import tensorboard_callback
import tf_keras as keras 
import tensorflow as tf

_SAVING_PATH = "/home/yassine/GitRepo/saved_models"

def create_base_EfficientNet_model(isTrainable = False):
    base_model = keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
    base_model.trainable = isTrainable
    return base_model

def create_EfficientNet_model_keras_api(train_data, test_data, saving_path = _SAVING_PATH, model_name="test_EfficientNet_model_keras_api",experiment_name='exp', image_shape = (224,224) ,isTrainable =False, number_class=10):
    base_model = create_base_EfficientNet_model(isTrainable)
    inputs = keras.layers.Input(shape=image_shape +(3,), name="input_layer")
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
    outputs = keras.layers.Dense(number_class, activation="softmax", name="output_layer")(x)
    model = keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])
    history = model.fit(train_data,
                                    epochs=5,
                                    steps_per_epoch=len(train_data),
                                    validation_data=test_data,
                                    validation_steps=int(0.25 * len(test_data)),
                                    callbacks=[tensorboard_callback(model_name,experiment_name,saving_path)])
    model.save(saving_path + '/'+model_name+'/model')
    return (model, history)
    
def inspect_layers(base_model):
    for layer_number, layer in enumerate(base_model.layers):
        print(layer_number, layer.name, layer.trainable )

# train_data_path_10_percent = "/home/yassine/env_tensorflow/GitRepo/Data/10_food_classes_10_percent/train"
# test_data_path_10_percent = "/home/yassine/env_tensorflow/GitRepo/Data/10_food_classes_10_percent/test"
# train_data_10_percent = data_loader(train_data_path_10_percent)
# test_data_10_percent = data_loader(test_data_path_10_percent)

# base_model = create_base_EfficientNet_model()
# inspect_layers(base_model)
