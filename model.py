# As suggested in the instroduction, NVIDIA's network architecture is a good candidate. 
# Therefore, this is the network I'm going to implement first. Once it is working, I'll 
# try out other networks and play with the real transfer learning aspect of the project.

import numpy as np
import cv2
import sklearn

from sklearn.model_selection import train_test_split
from random import shuffle

def load_driving_logs(data_path):
    context = []
    with open(data_path + "/driving_log.csv") as f:
        context = [line.strip() for line in f.readlines()]
    # in place shuffle which returns None.
    shuffle(context)
    return context

def flip_image_and_measurement(image, measurement):
    reversed_image = cv2.flip(image, 1)
    reversed_measurement = -1 * measurement
    return reversed_image, reversed_measurement

def process_image(data_path, image_path):
    # The following code read in the image as BGR! Convert it to RGB since the drive.py takes in RGB.
    image = cv2.imread(data_path + "/IMG/" + image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # add in flipping if necessary
    return image

def generator(samples, batch_size = 128, non_center_image_angle_correction = 0.2):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []

            for line in batch_samples:
                (center, left, right, steering_angle, throttle, brake, speed) = line.split(",")
                center_image = process_image(data_path, center.split("/")[-1])
                images.append(center_image)
                measurements.append(float(steering_angle))
                
                left_image = process_image(data_path, left.split("/")[-1])
                images.append(left_image)
                measurements.append(float(steering_angle) + non_center_image_angle_correction)

                right_image = process_image(data_path, right.split("/")[-1])
                images.append(right_image)
                measurements.append(float(steering_angle) - non_center_image_angle_correction)

            # Flipping the images
            reversed_images = []
            reversed_measurements = []

            for ind in range(len(images)):
                reversed_image, reversed_measurement = flip_image_and_measurement(images[ind], measurements[ind])
                reversed_images.append(reversed_image)
                reversed_measurements.append(reversed_measurement)

            images.extend(reversed_images)
            measurements.extend(reversed_measurements)

            X = np.array(images)
            y = np.array(measurements)
            yield sklearn.utils.shuffle(X, y)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

def construct_preprocessing_layers():
    model = Sequential()
    # Adding a normalization and mean-centering layer. 
    # This will first normalize the input to between 0 and 1, then it will center the mean to 0.
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    # Adding a cropping layer that remove the sky, trees and the front of the car.
    # As specified in the documentation (https://keras.io/layers/convolutional/#cropping2d), 
    # if tuple of 2 tuples of 2 ints: interpreted as ((top_crop, bottom_crop), (left_crop, right_crop))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))

    return model


def construct_nVidia_network():
    model = construct_preprocessing_layers()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu')) # here subsample means stride
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation = 'relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1))
    return model


data_path = "./data/"
samples = load_driving_logs(data_path)
print(len(samples))
train_sample, validation_sample = train_test_split(samples, test_size = 0.2)

train_generator = generator(train_sample, batch_size=64, non_center_image_angle_correction=0.2)
validation_generator = generator(validation_sample, batch_size=64, non_center_image_angle_correction=0.2)

model = construct_nVidia_network()
model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_sample), 
    validation_data = validation_generator, nb_val_samples = len(validation_sample), nb_epoch = 3)

model.save("model.h5")
# print("Train the network....")
# train_and_save_network(network, X_train, y_train)
# print("Training completed.")

import gc; gc.collect() 
