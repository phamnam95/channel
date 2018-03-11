
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import math
# Config the matlotlib backend as plotting inline in IPython
get_ipython().magic(u'matplotlib inline')
FLAGS = tf.app.flags.FLAGS

# Only 2 classes: channel or not-channel.
NUM_CLASSES = 2



tf.reset_default_graph()

def prelu(x, scope, decoder=False):
    '''
    Performs the parametric relu operation. This implementation is based on:
    https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
    For the decoder portion, prelu becomes just a normal prelu
    INPUTS:
    - x(Tensor): a 4D Tensor that undergoes prelu
    - scope(str): the string to name your prelu operation's alpha variable.
    - decoder(bool): if True, prelu becomes a normal relu.
    OUTPUTS:
    - pos + neg / x (Tensor): gives prelu output only during training; otherwise, just return x.
    '''
    #If decoder, then perform relu and just return the output
    if decoder:
        return tf.nn.relu(x, name=scope)

    alpha= tf.get_variable(scope + 'alpha', x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg

def msra_initializer(ker, fea):
    
    stddev = math.sqrt(2.0 / (ker**2 * fea))
    return tf.truncated_normal_initializer(stddev=stddev)




def orthogonal_initializer(scale=1.1):
    
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        flat_shape = (shape[0], np.prod(shape[1:]))  # 1st dim is batch size
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # Pick the one with right shape
        q = q.reshape(shape).astype(np.float32)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer    




def compute_loss(logits, labels):
    logits = tf.reshape(logits, [-1, NUM_CLASSES])
    labels = tf.reshape(labels, [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name="cross_entropy_per_example")
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'))




def batch_norm_layer(input_tensor, is_training, scope):
    return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(input_tensor, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn", reuse = False),
          lambda: tf.contrib.layers.batch_norm(input_tensor, is_training=False,
updates_collections=None, center=False, scope=scope+"_bn", reuse = True))
    




def my_conv3d(input_tensor, shape, keep_prob, activation=True, name=None, is_training=True):
    _, _, _,_, out_channel = tuple(shape)

    with tf.variable_scope(name) as scope:
        tf.get_variable_scope().reuse == True
        kernel = tf.get_variable(
            name="w", shape=shape,
            initializer=orthogonal_initializer())
        
        conv = tf.nn.conv3d(input_tensor, kernel, [1,1,1,1,1], padding='SAME')
        biases = tf.get_variable(
            name='biases', shape=[out_channel],
            initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        bias = tf.nn.dropout(bias, keep_prob)
        conv_out = batch_norm_layer(bias, is_training, scope.name)
        if activation is True:
            conv_out=tf.nn.relu(conv_out)
        

    return conv_out





def get_deconv_filter(shape):
    
    width, height, depth, in_channel, out_channel = tuple(shape)
    f = math.ceil(width/2.0)
    c = (2*f-1-f%2) / (2.0*f)
    bilinear = np.zeros([width, height, depth])
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                bilinear[x, y, z] = (1 - abs(x/f-c)) * (1 - abs(y/f-c)) * (1 - abs(z/f-c))
    weights = np.zeros(shape)
    for i in range(in_channel):
        for j in range(out_channel):
            weights[:,:,:, i, j] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name="up_filter",
                           initializer=init, shape=weights.shape)




def my_deconv3d(input_tensor, shape, output_shape, stride=2, name=None):
    strides = [1, stride,stride,stride, 1]
    with tf.variable_scope(name): 
        tf.get_variable_scope().reuse == True
        weights = get_deconv_filter(shape)
        deconv = tf.nn.conv3d_transpose(
            input_tensor, weights, output_shape=output_shape,
            strides=strides, padding='SAME')
       

    return deconv

def weighted_loss(logits, labels, head=None):
    """ median-frequency re-weighting """
    label_flatten = tf.reshape(labels, [-1])
    label_onehot = tf.one_hot(label_flatten, depth=2)
    logits_reshape = tf.reshape(logits, [-1, 2])
    count1=0
    count0=0
    frequency=np.array([0.6,0.4])
            
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=label_onehot, logits=logits_reshape,
                                                             pos_weight=frequency)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'))



def inference(x, y, keepprob, is_training):
    sz_height = 156
    sz_width = 156
    sz_depth = 100
    batch_size = tf.shape(x)[0]
    with tf.name_scope("Convnet1"):
        
        images = x
        images = tf.expand_dims(x, -1)
        
        conv1 = my_conv3d(images, [3,3,3, 1, 16], keep_prob=1, is_training=is_training, name="convonet1")
        

        print(conv1)

    with tf.name_scope("Maxpool1"):
        pool1 = tf.nn.max_pool3d(
            conv1, ksize=[1, 2,2,2, 1], strides=[1, 2,2,2, 1],
            padding='SAME', name="mpool1")
        

        print(pool1)

    sz2_height = sz_height // 2;
    sz2_width = sz_width//2;
    sz2_depth = sz_depth//2;
    with tf.name_scope("Convnet2"):
        conv2 = my_conv3d(pool1, [3,3,3, 16, 16], keep_prob=1, is_training=is_training, name="convonet2")
        

        print(conv2)

    with tf.name_scope("Maxpool2"):
        pool2 = tf.nn.max_pool3d(
            conv2, ksize=[1, 2,2,2, 1], strides=[1, 2,2,2, 1],
            padding='SAME', name="mpool2")
        

        print(pool2)

    sz3_height = sz2_height // 2;
    sz3_width = sz2_width//2;
    sz3_depth = sz2_depth//2;
    with tf.name_scope("Convnet3"):
        conv3 = my_conv3d(pool2, [3,3,3, 16, 16], keep_prob=1, is_training=is_training, name="convonet3")
        

        print(conv3)

    with tf.name_scope("Maxpool3"):
        pool3 = tf.nn.max_pool3d(
            conv3, ksize=[1, 2,2,2, 1], strides=[1, 2,2,2, 1],
            padding='SAME', name="mpool3")
        

        print(pool3)
        
    sz4_height = sz3_height // 2+1;
    sz4_width = sz3_width//2+1;
    sz4_depth = sz3_depth//2+1;
    with tf.name_scope("Convnet4"):
        conv4 = my_conv3d(pool3, [3,3,3, 16, 16], keep_prob=1, is_training=is_training, name="convonet4")
        

        print(conv4)

    with tf.name_scope("Maxpool4"):
        pool4 = tf.nn.max_pool3d(
            conv4, ksize=[1, 2,2,2, 1], strides=[1, 2,2,2, 1],
            padding='SAME', name="mpool4")
        

        print(pool4)
        
    
    pool4 = tf.nn.dropout(pool4, keep_prob=keepprob)
    
    
    with tf.name_scope("Deconv4"):
        upsamp4 = my_deconv3d(pool4, [2,2,2, 16, 16],
                              [batch_size, sz4_height,sz4_width,sz4_depth, 16],
                              stride=2, name="upsamp4")
        

        print(upsamp4)
        conv_decode4 = my_conv3d(upsamp4, [3,3,3, 16, 16],
                                 activation=False, keep_prob=1, is_training=is_training, name="conv_decode4")
        

        print(conv_decode4)
    
    
    
    with tf.name_scope("Deconv3"):
        
        upsamp3 = my_deconv3d(conv_decode4, [2,2,2, 16, 16],
                              [batch_size, sz3_height,sz3_width,sz3_depth, 16],
                              stride=2, name="upsamp3")
        

        print(upsamp3)
        conv_decode3 = my_conv3d(upsamp3, [3,3,3, 16, 16],
                                 activation=False,keep_prob=1, is_training=is_training, name="conv_decode3")
        

        print(conv_decode3)

    with tf.name_scope("Deconv2"):
        
        upsamp2 = my_deconv3d(conv_decode3, [2,2,2, 16, 16],
                              [batch_size, sz2_height,sz2_width,sz2_depth, 16],
                              stride=2, name="upsamp2")
      

        print(upsamp2)
        conv_decode2 = my_conv3d(upsamp2, [3,3,3, 16, 16],
                                 activation=False,keep_prob=1, is_training=is_training, name="conv_decode2")
        

        print(conv_decode2)

    with tf.name_scope("Deconv1"):
        upsamp1 = my_deconv3d(conv_decode2, [2,2,2, 16, 16],
                              [batch_size, sz_height,sz_width,sz_depth, 16],
                              stride=2, name="upsamp1")
       

        print(upsamp1)
        conv_decode1 = my_conv3d(upsamp1, [3,3,3, 16, 16],
                                 activation=False,keep_prob=1, is_training=is_training, name="conv_decode1")
        

        print(conv_decode1)
        
    with tf.name_scope("Classifier"):
        kernel = tf.get_variable(
            name="classifier_weights", shape=[1,1,1, 16, NUM_CLASSES],
            initializer=msra_initializer(1, 16))
        conv = tf.nn.conv3d(conv_decode1, kernel, [1,1,1,1,1], padding='SAME')
        print(conv)
        biases = tf.get_variable(
            name='classifier_biases', shape=[NUM_CLASSES],
            initializer=tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(conv, biases)
        print(logits)

    loss = compute_loss(logits, y)

    return loss, logits

    


