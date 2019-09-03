#!/usr/bin/env python3
import tensorflow as tf

class Model:

    def __init__(self, batch_size, actions, input_shape, learning_rate):
        self.batch_size = batch_size
        self.actions = actions
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.initialize = None
        self.construct_graph()
    
    def placeholder(self, shape, dtype):
        return tf.placeholder(shape=shape, dtype=dtype)

    def weights(self, shape, stddev):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
    
    def bias(self, const, shape):
        return tf.Variable(tf.constant(const, shape=shape))
    
    def hidden_conv_layer(self, layer, weights, strides, padding, bias):
        return tf.nn.relu(tf.nn.conv2d(layer, weights, strides=strides, padding=padding) + bias)
    
    def flatten_layer(self, layer):
        return tf.contrib.layers.flatten(layer)
    
    def fully_connected_layer(self, layer, units, activation):
        if activation is not None:
            return tf.layers.dense(inputs=layer, units=units, activation=activation)
        else:
            return tf.layers.dense(inputs=layer, units=units)
        
    def construct_graph(self):
        # input placeholder
        x = self.placeholder(shape=[None, 80, 80, 4], dtype=tf.float32)
        # convolutional layer 1
        W_conv1 = self.weights(shape=[8, 8, 4, 32], stddev=0.01)
        b_conv1 = self.bias(const=0.01, shape=[32])
        h_conv1 = self.hidden_conv_layer(layer=x, weights=W_conv1, strides=[1, 4, 4, 1], padding='SAME', bias=b_conv1)
        # convolutional layer 2
        W_conv2 = self.weights(shape=[4, 4, 32, 64], stddev=0.01)
        b_conv2 = self.bias(const=0.01, shape=[64])
        h_conv2 = self.hidden_conv_layer(layer=h_conv1, weights=W_conv2, strides=[1, 2, 2, 1], padding='SAME', bias=b_conv2)
        # convolutional layer 3
        W_conv3 = self.weights(shape=[3, 3, 64, 64], stddev=0.01)
        b_conv3 = self.bias(const=0.01, shape=[64])
        h_conv3 = self.hidden_conv_layer(layer=h_conv2, weights=W_conv3, strides=[1, 1, 1, 1], padding='SAME', bias=b_conv3)
        # dense layer 1
        h_conv3_flat = self.flatten_layer(layer=h_conv3)
        h_fc1 = self.fully_connected_layer(layer=h_conv3_flat, units=1024, activation=tf.nn.relu)
        # dense layer 2
        readout = self.fully_connected_layer(layer=h_fc1, units=self.actions)
        # predicted action
        predict = tf.argmax(readout, axis=1)
        # action variables
        a = self.placeholder(shape=[self.batch], dtype=tf.uint8) # actions
        a_one_hot = tf.one_hot(a, self.actions, dtype=tf.float32) #one-hot-encoded actions
        y = self.placeholder(shape=[self.batch], dtype=tf.float32) # target rewards
        q = tf.reduce_sum(tf.multiply(readout, a_one_hot), axis=1)
        # loss function and optimizer
        loss = tf.reduce_mean(tf.squared_difference(y, q))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.inititialize = tf.global_variables_initializer()