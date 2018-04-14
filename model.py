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
		(center, left, right, steering_angle, m2, m3, speed) = line.split(",")
		print("Processing image " + center)
		center_image = process_image(center)
		images.append(center_image)
		measurements.append(steering_angle)
		
		# left_image = process_image(data_path, left)
		# images.append(left_image)
		# measurements.append(steering_angle + non_center_image_angle_correction)

		# right_image = process_image(data_path, right)
		# images.append(right_image)
		# measurements.append(steering_angle - non_center_image_angle_correction)

	# converting to numpy array to ease the process down the pipeline.
	return (np.array(images), np.array(measurements))

def process_image(image_path):
	# The following code read in the image as BGR! Convert it to RGB since the drive.py takes in RGB.
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# add in flipping if necessary

	return image

preprocess_images_and_measurements("/Users/damao/Desktop/car data/")
