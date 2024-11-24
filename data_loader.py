import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_images_and_labels(directory, img_height, img_width, exclude_classes=set()):
    images = []
    labels = []
    class_names = sorted(set(os.listdir(directory)) - exclude_classes)
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(class_indices[class_name])

    images = np.array(images, dtype=np.float16)
    labels = np.array(labels)

    return images, labels, class_names
