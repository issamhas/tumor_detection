# Cancer Detection using Machine Learning

Used python and tensorflow to implement alexnet to classify mammogram images as normal, benign or malignant.


## Dataset
It was very hard to find a good quality dataset that contained more than 50 images. I was fortunate to find a subset of the
MIAS dataset that contained over 300 images, with an accompanying label text file.

The data was obtained from http://peipa.essex.ac.uk/info/mias.html

## Loading Dataset

To load the dataset I had to convert the images from .pgm format to .jpg using Pillow/PIL. Since the images were greyscale
the images were represented by 2d numpy array's containing values from 0 to 255, where 0 is black and 255 is white.

The labels were included in a text file, which I parsed to get label data, and then stored in a numpy array.

## Approach

I chose a supervised machine learning approach to classify the mammogram images, as I was fortunate to find a database of over 
300 images which were appropriately labeled. 

I chose to use convolutional neural networks as the basis for building my models. CNNs have been shown to be the best method to
classify images. This has been proven in the ImageNet competitions of the past few years.

Reference of 2017 results: http://image-net.org/challenges/LSVRC/2017/results

I chose to use tensorflow over other libraries such as scikitlearn and keras because it allows for more control and hyperparameter
tuning.

There are two .py files available in the repo, the softmax was used as a learning aid to understand how to import the MIAS dataset 
into tensorflow. Once that was finished I implemented a slightly modified version of the alexnet in the alexnet.py folder. 
The main difference is for the normalization I used tensorflow's local response normalization, instead of other methods such as 
batch_normalization. This is because it required the least tuning of parameters and provided default values for the normalization.

## Results and Steps for improvement

The alexnet model gave a prediction accuracy of 78% on 82 testing images. The steps taken to imrove prediction results would include:
1. Preprocessesing images to better identify areas of concern
2. Artificially expanding the dataset to have more training images
3. Experimenting with different algorithms/models and spend more time tuning hyperparameters.

## Steps to Run

If you'd like to run the program, feel free to download my source code. The repo doesn't contain the dataset so you would need to 
go to http://peipa.essex.ac.uk/info/mias.html to download it. 
