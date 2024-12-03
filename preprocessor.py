import gc
import os
import sys
import numpy as np
import tensorflow as tf
from typing import Literal
from sklearn.model_selection import train_test_split


def mem_check():
    def get_size(obj):
        return sys.getsizeof(obj)

    variables = globals()

    sorted_vars = sorted(
        variables.items(), key=lambda item: get_size(item[1]), reverse=True
    )

    total_memory = 0
    for var_name, var_value in sorted_vars[:10]:
        size = get_size(var_value)
        total_memory += size
        print(f"{var_name}: {size / (1024 ** 2):.2f} MB")

    print(f"Total memory usage: {total_memory / (1024 ** 2):.2f} MB")


def load_images_and_labels(directory, img_height, img_width, exclude_classes):
    images = []
    labels = []
    class_names = sorted(set(os.listdir(directory)) - exclude_classes)
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


def augment_images(images, labels, class_names):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    unique_classes, class_counts = np.unique(labels, return_counts=True)
    max_count = np.max(class_counts)

    balanced_images = []
    balanced_labels = []

    for class_id in unique_classes:
        print("Balancing class", class_names[class_id])
        # get all images and labels of the current class
        class_images = images[labels == class_id]
        class_labels = labels[labels == class_id]

        # add the ones we already have to the balanced dataset
        balanced_images.extend(class_images)
        balanced_labels.extend(class_labels)

        num_to_generate = max_count - len(class_images)

        if num_to_generate > 0:
            augmented_images = []
            augmented_labels = []
            # generate new images
            for x_batch, y_batch in datagen.flow(
                class_images, class_labels, batch_size=num_to_generate
            ):
                augmented_images.extend(x_batch)
                augmented_labels.extend(y_batch)
                if len(augmented_images) >= num_to_generate:
                    break

            # add generated images to balanced dataset
            balanced_images.extend(augmented_images[:num_to_generate])
            balanced_labels.extend(augmented_labels[:num_to_generate])

    balanced_images = np.array(balanced_images, dtype=np.float16)
    balanced_labels = np.array(balanced_labels)
    balanced_images = balanced_images / 255.0  # normalize

    mem_check()
    print("Clearing memory...")

    del class_images
    del class_labels
    del augmented_images
    del augmented_labels
    del x_batch
    del y_batch
    del images
    del labels
    gc.collect()
    mem_check()

    return balanced_images, balanced_labels


def get_training_data(
    model: Literal["cnn", "svc"],
    directory,
    img_height,
    img_width,
    exclude_classes=set(),
):
    if model not in ["cnn", "svc"]:
        raise ValueError("Invalid model type. Choose 'cnn' or 'svc'.")

    images, labels, class_names = load_images_and_labels(
        directory, img_height, img_width, exclude_classes
    )

    balanced_images, balanced_labels = augment_images(images, labels, class_names)

    del images
    del labels
    gc.collect()

    print("Splitting training and testing data...")
    # CNN
    if model == "cnn":
        x_train, x_test, y_train, y_test = train_test_split(
            balanced_images,
            balanced_labels,
            test_size=0.2,
            stratify=balanced_labels,
            random_state=123,
        )

        del balanced_images
        del balanced_labels
        return x_train, x_test, y_train, y_test, class_names

    # SVC
    n_samples = balanced_images.shape[0]
    print("Number of samples and image shape:", n_samples, balanced_images.shape)
    # the images need to be flattened for SVC because it expects a 2D array. dont need to flatten for CNN because they can be trained on multi-dimensional data
    data = balanced_images.reshape((n_samples, -1))

    x_train, x_test, y_train, y_test = train_test_split(
        data, balanced_labels, test_size=0.2, stratify=balanced_labels, random_state=123
    )

    del balanced_images
    del balanced_labels
    return x_train, x_test, y_train, y_test, class_names
