#!/usr/bin/env python
import os
import numpy as np
import theano  
import lasagne  
from lasagne import layers  
from lasagne.updates import nesterov_momentum  
from nolearn.lasagne import NeuralNet  
from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix  
from scipy.ndimage import imread

from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer

from lasagne.layers import set_all_param_values

import skimage.transform

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32))
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp1'] = ConvLayer(
        net['conv1'], num_filters=160, filter_size=1, flip_filters=False)
    net['cccp2'] = ConvLayer(
        net['cccp1'], num_filters=96, filter_size=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['cccp2'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['drop3'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv2'] = ConvLayer(net['drop3'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp3'] = ConvLayer(
        net['conv2'], num_filters=192, filter_size=1, flip_filters=False)
    net['cccp4'] = ConvLayer(
        net['cccp3'], num_filters=192, filter_size=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['cccp4'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['drop6'] = DropoutLayer(net['pool2'], p=0.5)
    net['conv3'] = ConvLayer(net['drop6'],
                             num_filters=192,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['cccp5'] = ConvLayer(
        net['conv3'], num_filters=192, filter_size=1, flip_filters=False)
    net['cccp6'] = ConvLayer(
        net['cccp5'], num_filters=9, filter_size=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['cccp6'],
                             pool_size=8,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['output'] = FlattenLayer(net['pool3'])

    return net


def load_dataset():
    numberOfData = 10
    #numberOfData = 16479
    channels = 3
    size = 32
    img = np.empty([numberOfData, channels, size, size], dtype=np.float)
    y = np.empty([numberOfData, 1], dtype=np.int8)

    # Read in training data
    with open('strain.csv', 'r') as f:
        it = 0
        for line in f:
            oneImg = imread("images/" + line.split(',')[0])
            oneImg = skimage.transform.resize(oneImg, (32, 32), preserve_range=True)
            img[it] = oneImg.reshape((channels, size, size))
            y[it] = int(line.split(',')[1])
            it += 1

    img /= 255
    return img, y

def load_test():
    numberOfData = 4
    #numberOfData = 2752
    channels = 3
    size = 32
    img = np.empty([numberOfData, channels, size, size], dtype=np.float)

    # Read in test data
    with open('stest.csv', 'r') as f:
        it = 0
        for line in f:
            oneImg = imread("testImages/" + line.strip())
            oneImg = skimage.transform.resize(oneImg, (32, 32), preserve_range=True)
            img[it] = oneImg.reshape((channels, size, size))
            it += 1

    img /= 255
    return img

if __name__ == '__main__':
    X, y = load_dataset()
    print("=====Finish reading=====")

    net0 = NeuralNet(  
    layers=[('input', layers.InputLayer),  
            ('conv2d1', layers.Conv2DLayer),  
            ('maxpool1', layers.MaxPool2DLayer),  
            ('conv2d2', layers.Conv2DLayer),  
            ('maxpool2', layers.MaxPool2DLayer), 
            ('conv2d3', layers.Conv2DLayer),  
            ('maxpool3', layers.MaxPool2DLayer),  
            ('dropout1', layers.DropoutLayer),  
            ('dense', layers.DenseLayer),  
            ('dropout2', layers.DropoutLayer),  
            ('output', layers.DenseLayer),  
            ],  
    # input layer  
    input_shape=(None, 3, 64, 64),  
    # layer conv2d1  
    conv2d1_num_filters=32,  
    conv2d1_filter_size=(5, 5),  
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,  
    conv2d1_W=lasagne.init.GlorotUniform(),    
    # layer maxpool1  
    maxpool1_pool_size=(2, 2),      
    # layer conv2d2  
    conv2d2_num_filters=32,  
    conv2d2_filter_size=(5, 5),  
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,  
    # layer maxpool2  
    maxpool2_pool_size=(2, 2), 
    # layer conv2d3  
    conv2d3_num_filters=32,  
    conv2d3_filter_size=(5, 5),  
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,  
    # layer maxpool3
    maxpool3_pool_size=(2, 2), 
    # dropout1  
    dropout1_p=0.5,      
    # dense  
    dense_num_units=256,  
    dense_nonlinearity=lasagne.nonlinearities.rectify,      
    # dropout2  
    dropout2_p=0.5,      
    # output  
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=9,  

    # optimization method params  
    update=nesterov_momentum,  
    update_learning_rate=0.01,  
    update_momentum=0.9,  
    max_epochs=10,  
    verbose=1,  
    )

    net0.fit(X, y.ravel())

    with open('model.pkl', 'rb') as f:
        params = pickle.load(f)

    net1 = build_model()
    set_all_param_values(net1.layers_.values(), params['param values'])

    print("=====Finish training=====")

    testX = load_test()
    predict = net0.predict(testX)

    with open('result.csv', 'w') as file:
        with open('stest.csv', 'r') as f:
            file.write("id,label\n")
            i = 0
            for line in f:
                file.write("%s,%d\n" %(line.strip(), predict[i]))
                i += 1

    print("=====Finish predicting=====")
    