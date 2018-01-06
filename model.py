from keras import layers
from keras import models

import cv2
import numpy as np


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # model.summary()

    return model


# def predict(img_path):
#     # read to numpy array
#     X = cv2.imread(img_path)
#     # resize
#     X = cv2.resize(X, (224, 224))
#     # expand to batch
#     X = np.expand_dims(X, axis=0)

#     # return model.predict(X)
#     return X
