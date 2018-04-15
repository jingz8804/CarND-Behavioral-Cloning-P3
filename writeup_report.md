# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* first-track.mp4 is the video of the car in autonomous driving mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](https://github.com/jingz8804/CarND-Behavioral-Cloning-P3/blob/master/model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My network is based on the model from the [nVidia Autonomous Car group](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). With enough data (collected and augmented), the car has successfully went through a full lap on the road. The network architecture can be found [here](https://github.com/jingz8804/CarND-Behavioral-Cloning-P3/blob/master/model.py#L77-L103).

#### 2. Attempts to reduce overfitting in the model

The model initially has shown sign of overfitting. To reduce the overfitting, more data was collected and fewer epochs were configured in the training. The model was trained and validated on different data sets to ensure that the model was not overfitting (code line [109-117](https://github.com/jingz8804/CarND-Behavioral-Cloning-P3/blob/master/model.py#L109-L117)). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer. The parameters stay at their default value.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. It includes the data provided by Udacity as well as my own driving data collected from the first track. All three images were used in the training.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try out a well-known network architecture and based on the result on the training/validation, we continue to refine the model to overcome either underfitting or overfitting.

My first step was to use a convolution neural network model similar to the nVidia network as it is a proven working network architecture. The only difference is that there is a cropping layer added to my network for preprocessing the data. 

The data used in the first a few trials were collected on the first track by myself using the simulator:
* 2 runs both clock and counter-clock wise.
* 1 run of recovery driving by driving the car to the side of the road and recording the driving back to the center of the road.

Based on this dataset, the car passed a few turns but eventually went off the trail at the dirty muddy section of the road. Looking at the training and validation loss, it was clear that the model was overfitting (since validation loss went up after a few epochs while the training loss remains low). I suspect that my driving was not so good on the simulator so I decided to redo the driving. Well, the model is only going to be as good as the driver who trained it.

To combat the overfitting, I collected more data by more carefullying driving the simulator for 2 laps and adding the dataset provided by Udacity. These data are further augmented to produce more training data. The retrained model with only 3 epochs is abled to go through a full lap without leaving the road. There was a place at the end of the bridge that I thought the car would go off the trail, but it eventually corrected itself. 

#### 2. Final Model Architecture

Below is a model summary (# of parameters and output shape calculation please refer to [CS231N](http://cs231n.github.io/convolutional-networks/#conv)):
```
Layer                       Output Shape           # of Parameters       Note
===================================================================================================================
Lambda_Layer                (None, 160, 320, 3)    0                     This layer normalizes the data.   
-------------------------------------------------------------------------------------------------------------------
Cropping2D_Layer            (None, 90, 320, 3)     0                     This layer crops of unnecessary pixels.  
-------------------------------------------------------------------------------------------------------------------
Convolution2D_Layer1        (None, 43, 158, 24)    3*5*5*24+24 = 1824    5x5 filter, 2x2 stride and 'valid' padding.
-------------------------------------------------------------------------------------------------------------------
Convolution2D_Layer2        (None, 20, 77, 36)     24x5x5x36+36 = 21636  Same as above.
--------------------------------------------------------------------------------------------------------------
Convolution2D_Layer3        (None, 8, 37, 48)      36x5x5x48+48 = 43248  Same as above.
--------------------------------------------------------------------------------------------------------------
Convolution2D_Layer4        (None, 6, 35, 64)      48x3x3x64+64 = 27712  3x3 filter, 1x1 stride and 'valid' padding.
--------------------------------------------------------------------------------------------------------------
Convolution2D_Layer5        (None, 4, 33, 64)      64x3x3x64+64 = 36928  Same as above.
--------------------------------------------------------------------------------------------------------------
Flatten_Layer               (None, 8448)           0                     Flatten the output.
--------------------------------------------------------------------------------------------------------------
Dense_Layer1                (None, 1164)           9,834,636             Fully connected layer.
--------------------------------------------------------------------------------------------------------------
Dense_Layer2                (None, 100)            1164x100+100 = 116500 Same as above.
--------------------------------------------------------------------------------------------------------------
Dense_Layer3                (None, 50)             100x50+50 = 5050      Same as above.
--------------------------------------------------------------------------------------------------------------
Dense_Layer4                (None, 10)             50x10+10 = 510        Same as above.
--------------------------------------------------------------------------------------------------------------
Dense_Layer5                (None, 1)              10x1+1 = 11           Same as above.
```

#### 3. Creation of the Training Set & Training Process

As described above, the dataset I used came from two sources:
* The dataset provided by Udacity.
* The dataset collected from driving the first track for 2 rounds with center lane driving.

This in total provided 15109 driving images and their measurements.

To augment the data sat, I also flipped images and angles thinking that this would mimic the driving in the counter-clock wise direction. I could drive it myself, but flipping actually made it work. 

Another augmentation was to include the images from the side cameras with steering angle measurement correction. For left images, we add positive correction and negative for the right images. This is based on the fact that the simulator uses the center image to decide the steering angle. Assuming that the center camera sees what the left camera sees, it should command the car to turn right a bit to get back to the center lane and vice versa for the right camera case. This augmentation helps the car to recover from the side of the road, similar to the effect of collecting recovery driving data ourselves.

After the data augmentation process, I had 15109 x 6 = 90654 number of data points. However, loading all the data actually caused memory issues since [the initial pipeline will convert the image into numpy array in floating point numbers](https://github.com/jingz8804/CarND-Behavioral-Cloning-P3/blob/master/model-old.py). This ate up more than 6 GB of memory alone and my EC2 instance cannot handle it. To fix this issue, the data was supplied to the model by [using the Python generator](https://github.com/jingz8804/CarND-Behavioral-Cloning-P3/blob/master/model.py#L33-L71).

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

As mentioned above, the previous trainings used 7 epochs and less data so it overfitted the data. This time we had more data and fewer epochs (3 epochs instead of 7). Some outputs of the training:

```
Epoch 1/3
12288/12087 [==============================] - 32s - loss: 0.0289 - val_loss: 0.0232
Epoch 2/3
12288/12087 [==============================] - 27s - loss: 0.0206 - val_loss: 0.0233
Epoch 3/3
12288/12087 [==============================] - 26s - loss: 0.0215 - val_loss: 0.0198
```
