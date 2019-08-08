import os, glob, argparse
import numpy as np
import tensorflow as tf
from rdn import rdn
from data import Data
from calculate import calculate_metrics
import cv2
from PIL import Image
import config as cfg

class Restorer(object):
    def __init__(self, net, data, args):
        self.net = net
        self.data = data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Set5', type=str)
    parser.add_argument('--image_form', default='.png', type=str)
    parser.add_argument('--weight_file', default='rdn.ckpt', type=str)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--eval', default=True, type=bool)
    args = parser.parse_args()

    scale = int(input('Input scale = ?\n'))
    net = rdn(True, scale)
    data = Data(args.dataset, args.image_form, scale)