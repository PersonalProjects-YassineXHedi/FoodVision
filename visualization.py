import os
import tensorflow as tf
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import random
import data_manager as dm
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import pandas as pd 

_SAVING_PATH = "/home/yassine/GitRepo/saved_models"

def plot_loss_curves(history):
    """
    Plots training and validation loss/accuracy curves from a Keras model history.

    Args:
        history (tensorflow.keras.callbacks.History): A history object obtained from model.fit().

    Returns:
        None: Displays the loss and accuracy curves.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(loss))

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()


def display_pred_and_image(model, img, class_names):
    """
    Displays an image alongside the model's predicted class and probability.

    Args:
        model (tensorflow.keras.Model): The trained model used for prediction.
        img (tensorflow.Tensor): A preprocessed image tensor.
        class_names (list): A list of class labels corresponding to model outputs.

    Returns:
        None: Displays the image with predicted label.
    """
    pred = model.predict(tf.expand_dims(img, axis=0))
    if len(pred[0]) > 1: 
        pred_class = class_names[pred.argmax()] 
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])] 
    
    plt.figure()
    plt.imshow(img/255.) 
    plt.title(f"pred: {pred_class}, prob: {pred.max():.2f}")
    plt.axis(False)
    plt.show()

def display_all_preds_and_images(images_path, model, class_names_path):
    """
    Loads all images from a specified directory, applies a trained model for prediction, 
    and displays each image with its predicted class.

    Args:
        images_path (str): The directory containing images to predict.
        model (tensorflow.keras.Model): The trained model used for prediction.

    Returns:
        None: Displays images with predicted labels.
    """
    custom_food_images = [ images_path +'/' + img_path for img_path in os.listdir(images_path)]
    class_names = dm.get_data_class_names(class_names_path)
    for img in custom_food_images:
        img = dm.load_and_prep_image(img, False) 
        display_pred_and_image(model,img,class_names)

def view_random_image(target_dir, target_class_name, view_single_image=True):
    """
    Displays a random image from a specified class directory.

    Args:
        target_dir (str): Path to the directory containing class folders.
        target_class (str): Name of the class to display an image from.

    Returns:
        np.ndarray: The selected image as a NumPy array.
    """

    target_folder = target_dir+'/'+target_class_name

    random_image = random.sample(os.listdir(target_folder), 1)

    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class_name)
    plt.axis("off")
    plt.text(0.5, -0.1, f"Image shape: {img.shape}", fontsize=12, ha="center", transform=plt.gca().transAxes)
    if(view_single_image):
        plt.show()

    return img



def view_random_images_from_classes(target_dir,target_class1,target_class2, target_class3 = None, target_class4 = None):
    """
    Displays 2, 3, or 4 random images from different class directories.

    Args:
        target_dir (str): Path to the directory containing class folders.
        target_class1 (str): Name of the first class folder.
        target_class2 (str): Name of the second class folder.
        target_class3 (str, optional): Name of the third class folder (default: None).
        target_class4 (str, optional): Name of the fourth class folder (default: None).

    Returns:
        None: Displays images in a subplot format.
    """
    if(target_class3 is None and target_class2 is None):
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        view_random_image(target_dir, target_class1, False)
        plt.subplot(1, 2, 2)
        view_random_image(target_dir, target_class2, False)
    else:
        plt.figure(figsize=(10,5))
        plt.subplot(2, 2, 1)

        view_random_image(target_dir, target_class1, False)
        plt.subplot(2, 2, 2)

        view_random_image(target_dir, target_class2, False)
        if(target_class3 is not None):
            plt.subplot(2, 2, 3)
            view_random_image(target_dir, target_class3, False)
        if(target_class4 is not None):
            plt.subplot(2, 2, 4)
            view_random_image(target_dir, target_class4, False)
    plt.tight_layout(pad=1)
    plt.show()


def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') 
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def create_confusion_matrix(y_true, y_pred, model_name,classes=[],saving_path=_SAVING_PATH, figsize=(9, 9), text_size=15, norm=True, savefig=False): 
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).
        norm: normalize values or not (default=False).
        savefig: save confusion matrix to file (default=False).
    
    Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.

    Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    """  
    plt.clf()  # **Clear any previous figures**
    plt.close('all')

    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with
    
    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if len(classes) > 0:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm_norm[i, j]*100:.1f}%",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > threshold else "black",
                    size=text_size-5)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > threshold else "black",
                    size=text_size)
    
    # Save the figure to the current working directory
    if savefig:
        fig.savefig(saving_path + '/'+model_name+"/confusion_matrix.png")
    plt.show()


def get_pred_data_table(data_path, data,pred_probs, y_labels,pred_classes, class_names):
    filepaths = []
    for filepath in data.list_files(data_path+ "/*/*.jpg", shuffle=False):
        filepaths.append(filepath.numpy())
    pred_df = pd.DataFrame({
    "img_path":filepaths,
    "y_true":y_labels,
    "y_pred":pred_classes,
    "pred_conf":pred_probs.max(axis=1),
    "y_true_classname": [class_names[i] for i in y_labels],
    "y_pred_classname": [class_names[i] for i in pred_classes]}) 
    pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
    return pred_df

def get_most_wrong_pred_table(data_path, data, pred_probs, y_labels, pred_classes, class_names, number_of_rows=100):
    pred_df = get_pred_data_table(data_path, data,pred_probs, y_labels,pred_classes, class_names)
    top_wrong = pred_df[pred_df["pred_correct"] == False].sort_values("pred_conf", ascending=False)[:number_of_rows]
    return top_wrong