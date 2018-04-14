# As suggested in the instroduction, NVIDIA's network architecture is a good candidate. 
# Therefore, this is the network I'm going to implement first. Once it is working, I'll 
# try out other networks and play with the real transfer learning aspect of the project.

import numpy as np
import cv2

# Before going right into building the network, we need to preprocess the data in our pipeline first.
def preprocess_images_and_measurements(data_path, non_center_image_angle_correction = 0.2):
    """
    Load in the image and measurement data we collected from the driving logs in the
    directory specified in the 'data_path'.

    The function will process the driving log one by one, also adding corrections to the non-center images.

    Here we are only interested in the images and their steering angle measurements. 

    This function returns a tuple of (images, measurements of the steering angle). 

    In the first trial, I'll only use the center image since we have data collected from the recovery lap. 
    Also no flipping of the image for now since I have collected images from reversed driving.
    """
    context = []
    with open(data_path + "/driving_log.csv") as f:
        context = [line.strip() for line in f.readlines()]

    images = []
    measurements = []

    # for testing purpose, only process the first 50 images.
    # context = context[:50]

    for line in context:
        (center, left, right, steering_angle, throttle, brake, speed) = line.split(",")
        # print("Processing image " + center)
        # the center contains the absolute path, which may not exist on remote host so we will need to extract only the filename
        # and supply a directory path which contains the file. 
        center_image = process_image(data_path, center.split("/")[-1])
        images.append(center_image)
        measurements.append(float(steering_angle))
        
        # left_image = process_image(data_path, left.split("/")[-1])
        # images.append(left_image)
        # measurements.append(float(steering_angle) + non_center_image_angle_correction)

        # right_image = process_image(data_path, right.split("/")[-1])
        # images.append(right_image)
        # measurements.append(float(steering_angle) - non_center_image_angle_correction)

    # converting to numpy array since that's the format Keras requires.
    return (np.array(images), np.array(measurements))

def process_image(data_path, image_path):
    # The following code read in the image as BGR! Convert it to RGB since the drive.py takes in RGB.
    image = cv2.imread(data_path + "/IMG/" + image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # add in flipping if necessary
    return image

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


X_train, y_train = preprocess_images_and_measurements("./data/")
model = construct_nVidia_network()
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7)

model.save("model.h5")
# print("Train the network....")
# train_and_save_network(network, X_train, y_train)
# print("Training completed.")

import gc; gc.collect() 
