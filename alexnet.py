#from softmax import test
from PIL import Image
import os, sys
import tensorflow as tf
import numpy as np
import csv
import cv2

#AlexNet Implementation

#constants
_epochs = 20
_weight_initialization = tf.contrib.layers.xavier_initializer()

#convolutional layer 1 parameters
num_neurons_c1 = 96
kernel_size_1 = 11
stride_length = 4

#max pooling layer 1
pool_size = 3


#convolutional layers with relu
def cnn_layer(input, filter, biases, stride_length):
	strides=[1,stride_length,stride_length,1]
	#filter is a 1d tensor that contains, filter height, width,
	#the number of input channels and output channels 
	
	#padding = 'SAME' identifies that the padding is the same size as the 
	#stride size
	input_weights = tf.nn.conv2d(input, filter, 
		strides, padding='SAME')
	y = tf.nn.bias_add(input_weights, biases)
	return tf.nn.relu(y)


def max_pool_layer(input, pool_size):
		kernels = [1,pool_size, pool_size, 1]
		strides = [1,pool_size, pool_size, 1]
		return tf.nn.max_pool(input, ksize=kernels, 
			strides=strides, padding='SAME')

#dense/fully connect layer
def dense_layer(input, weights, biases):
	y = tf.add(tf.matmul(input,weights), b)
	return tf.nn.relu(y)



	