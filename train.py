import os, glob, argparse, time, shutil

import numpy as np
import tensorflow as tf

from rdn import rdn
from data import Data
import config as cfg

class Recover(object):
    def __init__(self, net, data, scale):
        self.net = net
        self.data = data
        self.scale = scale
        self.image_size = cfg.IMAGE_SIZE
        self.label_size = cfg.LABEL_SIZE
        self.epochs = cfg.EPOCHS

        self.learning_rate = cfg.L_R
        self.global_step = tf.Variable(0, trainable=False)
        
        variables_to_save = tf.global_variables()
        variables_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(variables_to_save, max_to_keep=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Set14', type=str)
    parser.add_argument('--image_form', default='.png', type=str)
    parser.add_argument('--fresh', default=False, type=bool)
    parser.add_argument('--weight_file', default='rdn.ckpt', type=str)
    args = parser.parse_args()

    if args.fresh:
        weights = glob.glob(os.path.join('weights', '*.ckpt*'))
        if len(weights) > 0:
            for weight in weights:
                os.remove(weight)
        events = glob.glob(os.path.join('cache' + os.sep + 'event', 'event*'))
        if len(events) > 0:
            for event in events:
                os.remove(event)
        print("freshing complete")
    
    scale = int(input('Input scale: ?\n'))
    
    net = rdn(True, scale)
    data = Data(args.dataset, args.image_form, scale)

    recover = Recover(net, data, scale)