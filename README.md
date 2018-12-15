
# Deep Learning for Human Activity Recognition

### Aim:

This project aims to develop a Convolution Network to perform activtiy recognintion of physical activities using on-body Inertial Measurement Sensors.
    
Tensorflow and Keras APIs were used for the development of a 1D Sequential CNN of 7 Layers.
    
An old version of the Actitracker dataset from the  Wireless Sensor Data Mining (WISDM) lab at Fordham University, NY, USA was used to train and test the network. The latest dataset can be downloaded from: http://www.cis.fordham.edu/wisdm/dataset.php#actitracker
    
The activity states provided in the dataset are jogging, walking, ascending stairs, descending stairs, sitting and standing. The data was collected from 36 users using a smartphone in their pocket with the 20Hz sampling rate (20 values per second).
     

### Pipeline

The training process starts by reading the data and normalizing it. This normalized data is then segmented into time slices of window size 80 which translates to 4 seconds long chunks of data. These chunks are then randomly split into training and test sets. For the results shown in this report, 70% data as taken into the test set and the remaining was used in the test set for validation of the training algorithm. This training data was fed to a 1D CNN network which is described below.


### This section plots one window size long plots for each class of the normalized data


![png](https://github.com/bharatm11/1D_CNN_Human_activity_recognition/blob/master/output_7_0.png)



![png](https://github.com/bharatm11/1D_CNN_Human_activity_recognition/blob/master/output_7_1.png)



![png](https://github.com/bharatm11/1D_CNN_Human_activity_recognition/blob/master/output_7_2.png)



![png](https://github.com/bharatm11/1D_CNN_Human_activity_recognition/blob/master/output_7_3.png)



![png](https://github.com/bharatm11/1D_CNN_Human_activity_recognition/blob/master/output_7_4.png)



![png](https://github.com/bharatm11/1D_CNN_Human_activity_recognition/blob/master/output_7_5.png)


### CNN Network

A 1D CNN network was used considering the dimensions of the data. Each row of the data consists of the x,y,z accelerations from the accelerometer and the height of the layer determines the number of instances of data equalling the window size which is 80 in our case. Only the size of the input and output layers needs to be specified explicitly. The netork estimates the size of the hidden layers on it's own.

The network used here is of sequential type which means that it's basically a stack of layers. These layers include:
* Input layer
* First 1D CNN Layer
* A max pooling layer
* Second 1D CNN Layer 
* An average pooling layer
* A dropout layer
* A fully connected Softmax Activated layer

**Input Layer:** The input data consists of 80 time slices long instances of 3-axis accelerometer. Hence, the size of the input layer needs to be reshaped to 80x3. The data passes through the input layer as a vector of length 240. The output for this layer is 80x3.

**First 1D CNN Layer:** This defines a filter of kernel size 10. 100 such filters are defined in this layer to enable it to learn 100 different features. **Input Layer:** The input data consists of 80 time slices long instances of 3-axis accelerometer. Hence, the size of the input layer needs to be reshaped to 80x3. The data passes through the input layer as a vector of length 240. The output for this layer is a 71x100 matrix of neurons where the weights of each filter are defined column-wise.

**A max pooling layer:** This is used to reduce the complexity of the output and to prevent overfitting of the data. Using a pooling layer size of 3 reduces the size of the output matrix to 1/3rd of the input matrix.

**Second 1D CNN Layer:** This layer enables the network to pick up higher level features which were missed in the First CNN layer. The output of this layer is a 14x160 matrix.

**Average pooling layer:** This averages the value of two weights in the network thereby further reducing overfitting. The output is 1x160 matrix of neurons.

**Dropout layer:** This randomly assignms a weight of 0 to the neurons in the network. A value of 0.5 indicates that 50% of the neurons turn 0.

**Fully connected Softmax Activated layer:** This reduces the output to the desired height of 6 which indicates the number of activity classes in the data. Softmax forces all six outputs of the neural network to sum up to one.



    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    reshape_1 (Reshape)          (None, 80, 3)             0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 71, 100)           3100      
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 23, 100)           0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 14, 160)           160160    
    _________________________________________________________________
    global_average_pooling1d_1 ( (None, 160)               0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 160)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 6)                 966       
    =================================================================
    Total params: 164,226
    Trainable params: 164,226
    Non-trainable params: 0
    _________________________________________________________________
    None

### Results

The network was succesfully trained to recognize human activities using data obtained from on-body 3-axis accelerometers. 

A test accuracy of 92.66 % and training accuracy of 93.73% was achieved. The algorithm converges in 11 epochs equalling approximately 46 seconds. 

### References

* Jeffrey W. Lockhart, Gary M. Weiss, Jack C. Xue, Shaun T. Gallagher, Andrew B. Grosner, and Tony T. Pulickal (2011). "Design Considerations for the WISDM Smart Phone-Based Sensor Mining Architecture," Proceedings of the Fifth International Workshop on Knowledge Discovery from Sensor Data (at KDD-11), San Diego, CA
* Gary M. Weiss and Jeffrey W. Lockhart (2012). "The Impact of Personalization on Smartphone-Based Activity Recognition," Proceedings of the AAAI-12 Workshop on Activity Context Representation: Techniques and Languages, Toronto, CA. 

* Jennifer R. Kwapisz, Gary M. Weiss and Samuel A. Moore (2010). "Activity Recognition using Cell Phone Accelerometers," Proceedings of the Fourth International Workshop on Knowledge Discovery from Sensor Data (at KDD-10), Washington DC.

* https://keras.io/getting-started/sequential-model-guide/
* http://aqibsaeed.github.io/2016-11-04-human-activity-recognition-cnn/
* https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf


### Dataset

Actitracker dataset from the  Wireless Sensor Data Mining (WISDM) lab at Fordham University, NY, USA was used to train and test the network. The latest dataset can be downloaded from: http://www.cis.fordham.edu/wisdm/dataset.php#actitracker
