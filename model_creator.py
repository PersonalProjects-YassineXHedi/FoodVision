# Environment Setup
import os 
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
os.environ["TF_DEVICE_ALLOCATION"] = "gpu"

#Importation
import tf_keras as keras
import tensorflow as tf
import tensorflow_hub as hub
import datetime

_SAVING_PATH = "/home/yassine/GitRepo/saved_models"

def tensorboard_callback(model_name, experiment_name, saving_path = _SAVING_PATH ):
    log_dir = os.path.join(saving_path, model_name, "tensorboard_callback", experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    return tensorboard_callback

def checkpoint_callback(model_name, experiment_name, saving_path = _SAVING_PATH):
    checkpoint_path = os.path.join(saving_path, model_name, "checkpoint_callback", experiment_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         save_best_only=True, 
                                                         save_freq="epoch",
                                                         verbose=1)
    return checkpoint_callback



def create_model_form_tensorflow_hub(model_url, train_data, test_data, saving_path = _SAVING_PATH, model_name="test_model",experiment_name='exp', image_shape = (224,224), num_classes = 10):
  """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.
  
  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in output layer,
      should be equal to number of target classes, default 10.

  Returns:
    An uncompiled Keras Sequential model with model_url as feature
    extractor layer and Dense output layer with num_classes outputs.
  """
  feater_extraction_layer = hub.KerasLayer(model_url,
                                  trainable=False,
                                  input_shape = image_shape+(3,))
  model = keras.Sequential([
              keras.layers.Rescaling(scale=1/255., input_shape=image_shape+(3,)),
              feater_extraction_layer,
              keras.layers.Dense(10, activation='softmax',name='output_ayer')])

  model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
  history = model.fit(train_data,
                epochs=5,
                steps_per_epoch=len(train_data),
                validation_data=test_data,
                validation_steps=len(test_data),
                callbacks=[tensorboard_callback(model_name,experiment_name,saving_path)])
  model.save(saving_path + '/'+model_name+'/model')
  return (model, history)


   
def create_model(train_data,test_data, saving_path=_SAVING_PATH, model_name="test_model1",experiment_name='exp', image_shape = (224,224)):
  
  data_augmented_layer = create_data_augmentation_keras_layer()
  model = keras.Sequential([
      keras.layers.InputLayer(input_shape=image_shape + (3,)),
      data_augmented_layer,
      keras.layers.Conv2D(10, 3, activation='relu'),
      keras.layers.MaxPool2D(),
      keras.layers.Conv2D(10, 3, activation='relu'),
      keras.layers.MaxPool2D(),
      keras.layers.Flatten(),
      keras.layers.Dense(10, activation='softmax')
      ])
  
  model.compile(loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"])

  history = model.fit(train_data, 
                      epochs=5,
                      steps_per_epoch=len(train_data),
                      validation_data=test_data,
                      validation_steps=len(test_data),
                      callbacks=[tensorboard_callback(saving_path,model_name,experiment_name)])
  
  model.save(saving_path + '/'+model_name+'/model')
  return (model, history)

def load_model(model_name, saving_path = _SAVING_PATH):
  model = keras.models.load_model(saving_path +'/'+ model_name + '/model')
  return model

def create_data_augmentation_keras_layer(rescale = True, image_shape = (224,224)):
  data_augmentation = keras.Sequential([
    keras.layers.InputLayer(image_shape +(3,)),
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2)
  ], name ="data_augmentation")
  if(rescale):
    data_augmentation.add(keras.layers.Rescaling(1./255))
  return data_augmentation


# #Test
# #Preparing data
# train_data_path = "/home/yassine/env_tensorflow/GitRepo/Data/10_food_classes_all_data/10_food_classes_all_data/train"
# test_data_path = "/home/yassine/env_tensorflow/GitRepo/Data/10_food_classes_all_data/10_food_classes_all_data/test"
# train_data = data_loader(train_data_path)
# test_data = data_loader(test_data_path)

# # resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
# # create_model_form_tensorflow_hub(resnet_url, train_data, test_data)

# #load Model
# model = load_model("test_model")
# model.evaluate(test_data)


# train_data_path_10_percent = "/home/yassine/env_tensorflow/GitRepo/Data/10_food_classes_10_percent/train"
# test_data_path_10_percent = "/home/yassine/env_tensorflow/GitRepo/Data/10_food_classes_10_percent/test"
# train_data_10_percent = data_loader(train_data_path_10_percent)
# test_data_10_percent = data_loader(test_data_path_10_percent)

# create_model(train_data_10_percent,test_data_10_percent)

