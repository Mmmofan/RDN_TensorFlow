import os, glob

from PIL import Image
import cv2
import tensorflow as tf
import config as cfg

class Data(object):
    def __init__(self, dataset, image_form, scale):
        self.scale = scale
        self.image_form = image_form
        self.dataset = dataset
        self.data_dir = cfg.DATA_DIR
        self.image_size = cfg.IMAGE_SIZE
        self.label_size = cfg.LABEL_SIZE
        self.channel = cfg.CHANNEL
        self.stride = cfg.STRIDE
        self.data = None

    def train_setup(self):
        print("No training file, making...")