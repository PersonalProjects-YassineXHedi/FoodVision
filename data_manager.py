import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import pathlib
import numpy as np
import tensorflow as tf
import tf_keras as keras
import matplotlib.pyplot as plt


def show_directory_content(path):
    """
    Walks through a specified directory and prints the number of subdirectories and files at each level.

    Args:
        path (str): The root directory path from which to start traversing.

    Returns:
        None: The function prints the results directly to the console.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}' ")

def get_data_class_names(dataPath):
    """
    Retrieves the names of class directories from a specified dataset path.

    Args:
        dataPath (str): The path to the dataset directory where class folders are stored.

    Returns:
        np.ndarray: A sorted array containing the names of class directories.

    """
    data_dir = pathlib.Path(dataPath )
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    return class_names


def data_loader(dirPath,shuffled = True):
    """
    Loads and preprocesses image data from a directory.

    Args:
        dirPath (str): The directory path containing images structured by class folders.
        rescale (bool, optional): Whether to rescale pixel values to [0,1] (default: True).

    Returns:
        tensorflow.data.Dataset
    """
    data = keras.utils.image_dataset_from_directory(dirPath,
                                                    image_size=(224, 224),
                                                    batch_size=32,
                                                    shuffle=shuffled,
                                                    label_mode='categorical')
    return data


def load_and_prep_image(filename, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Args:
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True

    Returns:
        tensorflow.Tensor: Preprocessed image tensor of shape (224, 224, 3).
  """
  img = tf.io.read_file(filename=filename) 
  img=tf.image.decode_jpeg(img, channels=3)
  img=tf.cast(img, tf.float32)
  img = tf.image.resize(img, [224, 224])
  if scale:
    return img/255.
  else:
    return img

def get_predClasses_yLabels_classNames(data_path,model):
    data = data_loader(data_path, shuffled = False)
    predicted_probabilities = model.predict(data)
    predicted_classes = predicted_probabilities.argmax(axis=1)
    class_names = get_data_class_names(dataPath=data_path)
    y_labels = []
    for images, labels in data.unbatch():
        y_labels.append(labels.numpy().argmax())
    return(predicted_probabilities, predicted_classes,y_labels,class_names)