import os
from glob import glob

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

SMILE = 1
NON_SMILE = 0


def load_dataset(source='./data', train_proportion=0.8):
    smile_images_file_path = os.path.join(source, 'SMILE_list.txt')
    non_smile_images_file_path = os.path.join(source, 'NON-SMILE_list.txt')

    smile_image_names = _load_image_names(smile_images_file_path)
    non_smile_image_names = _load_image_names(non_smile_images_file_path)

    X = []
    y = []
    faces_files = glob(os.path.join(source, 'lfwcrop_color', 'faces', '*.ppm'))
    for image_path in faces_files:
        image_name = image_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]

        if image_name in smile_image_names:
            image = _load_image_as_numpy(image_path)
            X.append(image)
            y.append(SMILE)
        elif image_name in non_smile_image_names:
            image = _load_image_as_numpy(image_path)
            X.append(image)
            y.append(NON_SMILE)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def _load_image_names(path):
    with open(path, 'r') as f:
        image_names = f.readlines()
        return [image_name.rsplit('.', 1)[0] for image_name in image_names]


def _load_image_as_numpy(path):
    image = Image.open(path)
    return np.array(image)
