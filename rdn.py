#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

import config as cfg

class rdn(object):
    def __init__(self, is_train=True):
        self.weights= {}
        self.biases = {}

        image_size = cfg.IMAGE_SIZE
        label_size = cfg.LABEL_SIZE
        self.channel = cfg.CHANNEL
        scale = cfg.SCALE
        
        self.D  = cfg.D
        self.C  = cfg.C
        self.G  = cfg.G
        self.G0 = cfg.G0

        self.init_param()
        self.input_ = tf.placeholder(tf.float32, [None, image_size, image_size, self.channel], name='input')
        self.batch  = tf.placeholder(tf.int8, [None], name='batch')
        self.label_ = tf.placeholder(tf.float32, [None, label_size, label_size, self.channel], name='label')
        self.output = self.build_net(self.input_)

        if is_train:
            self.loss = self.loss_layer(self.output, self.label_)
        print("RDN net build...")
        
    def init_param(self):
        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        channel = self.channel
        # param for sfe
        self.weights['w1'] = tf.Variable(tf.random_normal([3, 3, channel, G0], stddev=0.01), name='w1')
        self.biases['b1'] = tf.Variable(tf.zeros([G0]), name='b1')
        self.weights['w2'] = tf.Variable(tf.random_normal([3, 3, G0, G], stddev=0.01), name='w2')
        self.biases['b2'] = tf.Variable(tf.zeros([G]), name='b2')
        # param for rdn block
        for d in range(D):
            for c in range(C):
                self.weights['w{}_{}'.format(d+3, c)] = tf.Variable(tf.random_normal([3, 3, G0+c*G, G], stddev=0.01), name='w{}_{}'.format(d+3, c))
                self.biases['b{}_{}'.format(d+3, c)] = tf.Variable(tf.zeros([G]), name='b{}_{}'.format(d+3, c))
            self.weights['w{}_{}'.format(d+3, c+1)] = tf.Variable(tf.random_normal([1, 1, G0+C*G, G], stddev=0.01), name='w{}_{}'.format(d+3, c+1))
            self.biases['b{}_{}'.format(d+3, c+1)] = tf.Variable(tf.zeros([G]), name='b{}_{}'.format(d+3, c+1))
        # param for dff

    def build_net(self, input_tensor):
        # SFE
        x_pass, x_1 = self.sfenet(input_tensor)
        x_2 = self.rdn_block(x_1)
        return x_2

    def sfenet(self, input_tensor):
        x_pass = self.conv_layer(input_tensor, 'w1', 'b1', False, 'sfe1')
        x_1    = self.conv_layer(x_pass, 'w2', 'b2', False, 'sfe2')
        return x_pass, x_1

    def rdn_block(self, input_tensor):
        D = self.D
        C = self.C
        G = self.G
        G0 = self.G0
        rdb_concat = []
        rdb_in = input_tensor

        for d in range(D):
            x = rdb_in
            for c in range(C):
                temp = self.conv_layer(x, 'w{}_{}'.format(d+3, c+3), 'b{}_{}'.format(d+3, c+3), True, 'rdn{}_{}'.format(d+3, c+3))
                temp = tf.concat([temp, x], axis=3)
                x = temp
            temp = self.conv_layer(x, 'w{}'.format(d+4), 'b{}'.format(d+4), True, 'rdn{}'.format(d+4))
            rdb_concat.append(temp)
        return tf.concat(rdb_concat, axis=3)

    def conv_layer(self, input_tensor, weight, bias, activation, name):
        """
        Do convolution + bias, if activation is True, add relu as activation function
        Args:
            input_tensor:
            weight: weight name, str
            bias: bias name, str
            activation: do activation or not, bool
            name: op name, str
        """
        weight = self.weights[weight]
        bias = self.biases[bias]
        out = tf.nn.conv2d(input_tensor, weight, [1, 1, 1, 1], padding='SAME', name=name) + bias
        if activation:
            out = tf.nn.relu(out)
        return out

    def loss_layer(self, output, label):
        diff = tf.abs(output - label)
        loss = tf.reduce_mean(diff)
        return loss

if __name__ == "__main__":
    net = rdn(True)
    print("Done")