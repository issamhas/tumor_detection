from PIL import Image
import os, sys
import tensorflow as tf
import numpy as np
import csv
import cv2

#constants
_imgPath = 'bc_photos/mdb'
_labelsPath = 'photo_labels.txt'
_imgDim = 1024
_imgSize = 1024*1024
_epochs = 20
_trainSetSize = 200
_testSetSize = 75
_stepSize = 0.05
_labelColumns = 3


def convert_to_jpg(path):

    for i in range(321):
        if i < 9:
            img_path = _imgPath+'00'+str(i+1)
        elif i<99:
            img_path = _imgPath+'0'+str(i+1)
        else:
            img_path = _imgPath+str(i+1)

        img = Image.open(img_path+'.pgm')
        img.save(img_path+".jpg")


#get list of filenames for tensorflow input
"""
def get_file_names(path):
    images = []
    for i in range(322):
        if i < 9:
            temp_path = path+'00'+str(i+1)
        elif i<99:
            temp_path = path+'0'+str(i+1)
        else:
			temp_path = path+str(i+1)

		img = Image.open(temp_path+'.jpg')
		img.load()

		data = np.asarray(img,dtype="float32')
		
		images.append(data)
		
	result = np.array(images)
	print result.shape
	return result
"""
# get images as numpy array
def get_numpy_imgs(path):
	images = []
	for i in range(322):
	#values 00 and 0 are added to paths to coordinate to the file names in the folder bc_photos
	#ex mdb001, mdb10, mdb100
		if i < 9:
			temp_path = path+'00'+str(i+1)
		elif i<99:
			temp_path = path +'0' +str(i+1)
		else:
			temp_path = path +str(i+1)
		
		img = Image.open(temp_path+'.jpg')
		img.load()
		data = np.asarray(img,dtype='float32')
		images.append(data)
	
	result = np.array(images)
	return result

def get_flatened_images(path):
	not_flat = get_numpy_imgs(path)
	list_flat = []
	
	for i in range(322):
		temp = not_flat[i].flatten()
		list_flat.append(temp)
	
	return np.array(list_flat)
	
		

def gen_labels(file_name):
	labels = []
	txt_file = csv.reader(open(file_name), delimiter=" ")
	#if a mammogram returns as NORMAL, assign a value 0, if it is Benign(B) or Malignant(M) assign value 1 or 2
	for s in txt_file:
		if s[2] =='NORM':
			labels.append(int(0))
		elif s[3] == 'B':
			labels.append(int(1))
		else:
			labels.append(int(2))
	
	np_labels = np.array(labels)
	#np.eye makes a value 3 to [0,0,1,0,0], essential one-hot encoding
	return np.eye(3)[np_labels]

def simple_net():

	#input placeholders for images and labels
	x =tf.placeholder(tf.float32, [None, _imgSize])
	y_ = tf.placeholder(tf.float32, [None,_labelColumns])

	#generate flattened 1024x1024 images and labels from 
	images = get_flatened_images(_imgPath)
	labels = gen_labels(_labelsPath)
	
	#Weights and biases initialization
	W = tf.Variable(tf.zeros([_imgSize,_labelColumns]))
	b = tf.Variable(tf.zeros([_labelColumns]))
	
	#output function from softmax "y = mx+b"
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	#measure error
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	
	#gradient descent with 0.05 step size to limit error
	train_step = tf.train.GradientDescentOptimizer(_stepSize).minimize(cross_entropy)
	
	#launch interactive session
	sess = tf.InteractiveSession()
	
	tf.global_variables_initializer().run()

	#train for number of _epochs	
	for i in range(_epochs):
		
		print('start training')
		#train with first 200 elements of images and labels
		sess.run(train_step, feed_dict={x: images[:_trainSetSize], y_: labels[:_trainSetSize]})
		
		print('trained')
		
	#list of boolean values, size of test set that represents number of 
	#correct/incorrect predictions
	prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	
	#get mean of values that are correct, result is decimal from 0 to 1
	acc_mean = tf.reduce_mean(tf.cast(prediction, tf.float32))
	
	#prediction accuracy from test data
	print(sess.run(acc_mean, feed_dict={x: images[-_testSetSize:], y_: labels[-_testSetSize:]}))



#print(gen_labels('photo_labels.txt').shape)
#print(get_flatened_images(_imgPath).shape)
simple_net()
#print(gen_labels('photo_labels.txt')[:10])
#print(get_flatened_images(_imgPath).shape)