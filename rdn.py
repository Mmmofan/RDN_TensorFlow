#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

import config as cfg

class rdn(object):
    def __init__(self, is_train=True):
        self.is_train = is_train
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

        self.init_param(scale)
        self.input_ = tf.placeholder(tf.float32, [None, image_size, image_size, self.channel], name='input')
        self.batch  = tf.placeholder(tf.int8, [], name='batch')
        self.label_ = tf.placeholder(tf.float32, [None, label_size, label_size, self.channel], name='label')
        self.output = self.build_net(self.input_, scale)

        if is_train:
            self.loss = self.loss_layer(self.output, self.label_)
        print("RDN net build...")
        
    def init_param(self, scale):
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
                self.weights['w{}_{}'.format(d+3, c)] = tf.Variable(tf.random_normal([3, 3, (1+c)*G, G], stddev=0.01), name='w{}_{}'.format(d+3, c))
                self.biases['b{}_{}'.format(d+3, c)] = tf.Variable(tf.zeros([G]), name='b{}_{}'.format(d+3, c))
            self.weights['w{}_{}'.format(d+3, c+1)] = tf.Variable(tf.random_normal([1, 1, (1+C)*G, G], stddev=0.01), name='w{}_{}'.format(d+3, c+1))
            self.biases['b{}_{}'.format(d+3, c+1)] = tf.Variable(tf.zeros([G]), name='b{}_{}'.format(d+3, c+1))
        # param for dff
        self.weights['w{}'.format(D+3)] = tf.Variable(tf.random_normal([1, 1, G*D, G0], stddev=0.01), name='w{}'.format(D+3))
        self.biases['b{}'.format(D+3)] = tf.Variable(tf.zeros([G0]), name='b{}'.format(D+3))
        self.weights['w{}'.format(D+4)] = tf.Variable(tf.random_normal([3, 3, G0, G0], stddev=0.01), name='w{}'.format(D+4))
        self.biases['b{}'.format(D+4)] = tf.Variable(tf.zeros([G0]), name='b{}'.format(D+4))
        # param for upnet
        self.weights['w{}'.format(D+5)] = tf.Variable(tf.random_normal([5, 5, G0, 64], stddev=0.01), name='w{}'.format(D+5))
        self.biases['b{}'.format(D+5)] = tf.Variable(tf.zeros([64]), name='b{}'.format(D+5))
        self.weights['w{}'.format(D+6)] = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=0.01), name='w{}'.format(D+6))
        self.biases['b{}'.format(D+6)] = tf.Variable(tf.zeros([32]), name='b{}'.format(D+6))
        self.weights['w{}'.format(D+7)] = tf.Variable(tf.random_normal([3, 3, 32, channel*scale*scale], stddev=np.sqrt(2/9/32)), name='w{}'.format(D+7))
        self.biases['b{}'.format(D+7)] = tf.Variable(tf.zeros([channel*scale*scale]), name='b{}'.format(D+7))
        self.weights['w{}'.format(D+8)] = tf.Variable(tf.random_normal([3, 3, channel, channel], stddev=np.sqrt(2/9/32)), name='w{}'.format(D+8))
        self.biases['b{}'.format(D+8)] = tf.Variable(tf.zeros([channel]), name='b{}'.format(D+8))

    def build_net(self, input_tensor, scale):
        # SFE
        x_pass, x_1 = self.sfenet(input_tensor)
        x_2 = self.rdn_block(x_1)
        x_3 = self.dff(x_2, x_pass)
        x_4 = self.upnet(x_3, scale)
        return x_4

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
                temp = self.conv_layer(x, 'w{}_{}'.format(d+3, c), 'b{}_{}'.format(d+3, c), True, 'rdn{}_{}'.format(d+3, c))
                temp = tf.concat([x, temp], axis=3)
                x = temp
            temp = self.conv_layer(x, 'w{}_{}'.format(d+3, c+1), 'b{}_{}'.format(d+3, c+1), False, 'rdn{}_{}'.format(d+3, c+1))
            rdb_concat.append(temp)
        return tf.concat(rdb_concat, axis=3)

    def dff(self, input_tensor1, input_tensor2):
        D = self.D
        x = self.conv_layer(input_tensor1, 'w{}'.format(D+3), 'b{}'.format(D+3), False, 'dff1')
        x = self.conv_layer(x, 'w{}'.format(D+4), 'b{}'.format(D+4), False, 'dff2')
        return tf.add(x, input_tensor2)

    def upnet(self, input_tensor, r):
        D = self.D
        x = self.conv_layer(input_tensor, 'w{}'.format(D+5), 'b{}'.format(D+5), True, 'upnet1')
        x = self.conv_layer(x, 'w{}'.format(D+6), 'b{}'.format(D+6), True, 'upnet2')
        x = self.conv_layer(x, 'w{}'.format(D+7), 'b{}'.format(D+7), False, 'upnet3')
        x = self.PS(x, r)
        x = self.conv_layer(x, 'w{}'.format(D+8), 'b{}'.format(D+8), False, 'upnet4')
        return x

    def PS(self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        # 在feature maps维上，分成3个Tensor，每个的shape应该是(batch_size,H,W, self.scale * self.scale)
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3)  # Do the concat RGB
        else:
            X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3)  # Do the concat RGB
        return X

    # NOTE: train with batch size
    def _phase_shift(self, I, r):
        """
        把最后一位放大的scale转到Height和weight上
        :param I:
        :param r:放大因子
        :return:
        """
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (-1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (-1, a * r, b * r, 1))

    # NOTE: test without batchsize
    def _phase_shift_test(self, I, r):
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (1, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, b, a*r, r
        X = tf.split(X, b, 0)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 1)  # bsize, a*r, b*r
        return tf.reshape(X, (1, a * r, b * r, 1))


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