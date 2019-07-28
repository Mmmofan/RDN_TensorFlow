import os, glob

from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import config as cfg
import h5py

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
        print("No training file, making h5 file for training...")
        scale = self.scale
        stride = self.stride
        self.data = self.prepare_data(os.path.join(self.data_dir, self.dataset), self.image_form)
        if len(self.data) >= 100:
            group = len(self.data) // 50
            for gp in range(group):
                data_list = self.data[group * 50 : (group+1) * 50]
                input_seq, label_seq = [], []
                for data in data_list:
                    input_, label_ = preprocess(data, scale)
                    # for x
                        # for y
                            # input, label
                    # input_seq.append(input)
                    # label_seq.append(label)
                # make_data(input_seq, label_seq, self.dataset)
        else:
            input_seq, label_seq = [], []
            for data in self.data:
                input_, label_ = preprocess(data, scale)


    def prepare_data(self, dataset, image_form):
        """
        Args:
            dataset: the data directory, str
            image_forom: image file type, str
        """
        if not os.path.exists(dataset):
            raise Exception("No such dataset")
        data_files = glob.glob(os.path.join(dataset, '*'+image_form))
        return data_files


def preprocess(path, scale):
    """
    Read image and make a pair of input and label
    """
    image = Image.open(path).convert('RGB')
    (width, height) = image.size()
    label_ = np.array(list(image.get_data())).astype(np.float32).reshape((height, width)) / 255

    new_height, new_width = int(height / scale), int(width / scale)
    scaled_img = image.resize((new_width, new_height), Image.ANTIALIAS)
    input_ = np.array(list(scaled_img.get_data())).astype(np.float32).reshape((new_height, new_width)) / 255

    return input_, label_

def make_data(input_seq, label_seq, dataset):
    save_path = os.getcwd() + os.sep + 'cache' + os.sep + dataset + '_train_X{}'.scale + '.h5'  # ./RDN/cahce/Set5_train_X3.h5
    if not os.path.exists(save_path):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('data', chunks=True, maxshape=[None])

if __name__ == "__main__":
    data = Data('data'+os.sep+'Set5', '.png', 2)
    data.train_setup()
        