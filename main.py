from PIL import Image
import os, sys
import tensorflow as tf
#constants
_imgPath = 'bc_photos/mdb'
#image = Image.open('bc_photos/mdb322.pgm')
#image.save('bc_photos/mdb322.jpg')
#image.delete('bc_photos/mdb010.jpg')

#os.remove('bc_photos/mdb005.jpg')

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
        #os.remove(img_path+'.pgm')
#only convert once.
#convert_to_jpg(_imgPath)

#get list of filenames for tensorflow input
def get_file_names(path):
    filenames = []
    for i in range(322):
        if i < 9:
            temp_path = path+'00'+str(i+1)
        elif i<99:
            temp_path = path+'0'+str(i+1)
        else:
            temp_path = path+str(i+1)

        filenames.append(temp_path+'.jpg')
    

    #print(filenames)

    return filenames

	# returns queue for holding filenames using tf
def get_queue():
	#get filenames 
	fns = get_file_names(_imgPath)
	return tf.train.string_input_producer(fns)

def decode():
#filenames queue from get queue
	filename_q	= get_queue()
	rdr = tf.WholeFileReader()
	filename, content = rdr.read(filename_q)
	#decoding images into a list of tensors, channels identifies if it's 0(default), 
	#1(grayscale), 3(RGB) image
	img = tf.image.decode_jpeg(content,channels=3)
	#cast images to tensors 
	img = tf.cast(img,tf.float32)
	
	resized_img = tf.image.resize_images(img, [1024,1024])
	
	#batching
	
	

decode()
