import os
import numpy as np
import tensorflow as tf


def load_images_and_labels(directory, img_height, img_width, num_classes=101):
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))[-num_classes:]
    class_indices = {
        class_name: idx for idx, class_name in enumerate(class_names)
    }  # {name: index}

    for class_name in class_names:
        class_dir = os.path.join(
            directory, class_name
        )  # caltech-101/101_ObjectCategories/class_name
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(
                    class_dir, img_name
                )  # caltech-101/101_ObjectCategories/class_name/img_name
                img = tf.keras.preprocessing.image.load_img(
                    img_path, target_size=(img_height, img_width)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                images.append(img_array)
                labels.append(class_indices[class_name])

    return np.array(images), np.array(labels), class_names
