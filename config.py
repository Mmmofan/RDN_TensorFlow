#-*- coding: utf-8 -*-
import os

IMAGE_SIZE = 32
SCALE = 2
LABEL_SIZE = IMAGE_SIZE * SCALE
STRIDE = IMAGE_SIZE - 1
CHANNEL = 3
BATCH = 16
D = 20
C = 6
G = 32
G0 = 64

EPOCHS = 200
L_R = 0.0001

WEIGHTS_DIR = 'weights'
CACHE_DIR = 'cache'
EVENT_DIR = os.path.join(CACHE_DIR, 'event')
DATA_DIR = 'data'