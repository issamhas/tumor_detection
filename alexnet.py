from softmax import get_flatened_images, gen_labels
from PIL import Image
import os, sys
import tensorflow as tf
import numpy as np
import csv
import cv2

#AlexNet Implementation
_labelsPath = 'photo_labels.txt'
_imgPath = 'bc_photos/mdb'
_imgDim = 1024
_inputSize = 1024*1024
_numClasses = 3
#constants
_epochs = 20
_weight_initializer = tf.contrib.layers.xavier_initializer()

#convolutional layer 1 parameters
num_neurons_c1 = 96
kernel_size_1 = 11
stride_length1 = 4

#convolutional layer 2 paramaters
num_neurons_c2 = 256
kernel_size_2 = 5
stride_length2 = 1

#convolutional layer 2 paramaters
num_neurons_c3 = 384
kernel_size_3 = 3
stride_length3 = 1

#convolutional layer 2 paramaters
num_neurons_c4 = 384
kernel_size_4 = 3
stride_length4 = 1

#convolutional layer 2 paramaters
num_neurons_c5 = 256
kernel_size_5 = 3
stride_length5 = 1

bias_conv_1 = tf.Variable(tf.zeros([num_neurons_c1]))

#max pooling layer 1
max_pool_size = 3

#classes output
_classes = 3
#dense layer 1
#dense layer weights
pool_sqr = int(1024/3)
dense_inputs = pool_sqr**2*num_neurons_c5

#conv_weights('conv1',kernel_size, num_neurons)
weights = {
#weights -> kernel size, num neurons for each convolutional layer
	'conv1': tf.get_variable('conv1', [kernel_size_1,kernel_size_1,1,num_neurons_c1], 
		initializer = _weight_initializer),
	'conv2': tf.get_variable('conv2', [kernel_size_2,kernel_size_2,num_neurons_c1,num_neurons_c2], 
		initializer = _weight_initializer),
	'conv3': tf.get_variable('conv3', [kernel_size_3,kernel_size_3,num_neurons_c2,num_neurons_c3], 
		initializer = _weight_initializer),
	'conv4': tf.get_variable('conv4', [kernel_size_4,kernel_size_4,num_neurons_c3,num_neurons_c4], 
		initializer = _weight_initializer),
	'conv5': tf.get_variable('conv5', [kernel_size_5,kernel_size_5,num_neurons_c4,num_neurons_c5], 
		initializer = _weight_initializer),	
		
	
	'dense1': tf.get_variable('dense1',[20736,81],initializer=_weight_initializer),
	'dense2': tf.get_variable('dense2',[20736,81],initializer=_weight_initializer),
	'out': tf.get_variable('out',[81,_classes],initializer=_weight_initializer)
}

biases = {
#convolution layer biases
	'biasc1': tf.Variable(tf.zeros([num_neurons_c1])),
	'biasc2': tf.Variable(tf.zeros([num_neurons_c2])),
	'biasc3': tf.Variable(tf.zeros([num_neurons_c3])),
	'biasc4': tf.Variable(tf.zeros([num_neurons_c4])),
	'biasc5': tf.Variable(tf.zeros([num_neurons_c5])),

	#dense layer biases
	'biasd1': tf.Variable(tf.zeros([81])),
	'biasd2': tf.Variable(tf.zeros([81])),
	
	'bias_out': tf.Variable(tf.zeros([_classes]))

}

#-------------------------- Placeholders --------------------------------------------#

x = tf.placeholder(tf.float32, [None, _inputSize])
y_ = tf.placeholder(tf.float32, [None, _numClasses])

#Inputs

images = get_flatened_images(_imgPath)
labels = gen_labels('photo_labels.txt')

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
			strides=strides, padding='VALID')

#dense/fully connect layer
def dense_layer(input, weights, biases):
	y = tf.add(tf.matmul(input,weights), biases)
	return tf.nn.relu(y)
	
def get_2d_images():
	get_numpy_imgs(_imgPath)

def alex_net(input, weights, biases):
	#images = get_flatened_images(_imgPath)
	images = tf.reshape(input, shape=[-1, _imgDim, _imgDim,1])
	
	#convolutional layer 1
	c1 = cnn_layer(images, weights['conv1'], biases['biasc1'], stride_length1)
	max_pool_1 = max_pool_layer(c1,max_pool_size)
	norm1 = tf.nn.local_response_normalization(max_pool_1)
	#convolutional layer 2
	c2 = cnn_layer(norm1, weights['conv2'], biases['biasc2'], stride_length2)
	max_pool_2 = max_pool_layer(c2,max_pool_size)
	norm2 = tf.nn.local_response_normalization(max_pool_2)
	
	#convolutional layers 3 to 5
	c3 = cnn_layer(norm2, weights['conv3'], biases['biasc3'], stride_length3)
	c4 = cnn_layer(c3,weights['conv4'], biases['biasc4'],stride_length4)
	c5 = cnn_layer(c4,weights['conv5'],biases['biasc5'],stride_length5)
	max_pool_3 = max_pool_layer(c5,max_pool_size)#pool sizes 1 to 3 are the same
	
	
	
	#dense layers, convert to 1d tensor
	pool_3_shape = max_pool_3.get_shape().as_list()
	flat = tf.reshape(max_pool_3, 
		[-1, pool_3_shape[1]*pool_3_shape[2]*pool_3_shape[3]])
	
	dense_layer_1 = dense_layer(flat, weights['dense1'], biases['biasd1'])
	dense_layer_2 = dense_layer(flat, weights['dense2'], biases['biasd2'])
	
	return tf.add(tf.matmul(dense_layer_2, weights['out']), biases['bias_out'])
	
_batch_size = 40

def gen_next_batch(k):
	if(k==0):
		return images[:_batch_size], labels[:_batch_size]
	else:
		return images[_batch_size*k:_batch_size*(k+1)], labels[_batch_size*k:_batch_size*(k+1)]

def train_model():
	_changeable_range = 6

	#mmodel building
	y = alex_net(x, weights, biases)
	
	#loss calculations and optimization
	cross_entropy =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
	adam_optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
	
	prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
	
	#launch interactive session
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	
	for i in range(_epochs):
		print("Starting training")
		for k in range(_changeable_range):
			print("Start minibatch "+str(k))
			image_btch, label_btch = gen_next_batch(k)
			_, btch_acc = sess.run([adam_optimizer, accuracy],feed_dict={x:image_btch ,y_:label_btch})
			print("End minibatch" +str(btch_acc))
	
		#_changeable_range = 3
	print("End training")
	#test_acc = accuracy.eval({x: images[-82:], y_:labels[-82:]})
	test_acc=sess.run(accuracy,feed_dict={x: images[-82:], y_:labels[-82:]})
	print(test_acc)
#print(alex_net(x,weights,biases))	
train_model()
	
