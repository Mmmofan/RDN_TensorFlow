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
        self.data = prepare_data(os.path.join(self.data_dir, self.dataset+os.path.sep+'image_SRF_2'), self.image_form)
        if len(self.data) >= 100:  # in case the data is too large
            group = len(self.data) // 50
            for gp in range(group):
                data_list = self.data[group * 50 : (group+1) * 50]
                input_seq, label_seq = [], []
                for data in data_list:
                    input_, label_ = preprocess(data, scale)
                    (height, width) = input_.shape
                    x_range = (height - self.image_size) // stride
                    y_range = (width - self.image_size) // stride
                    for x in range(x_range):
                        for y in range(y_range):
                            input_patch = input_[x * stride : x * stride + self.image_size,
                                                 y * stride : y * stride + self.image_size, :]
                            label_patch = label_[x * stride * scale : x * stride * scale + self.label_size,
                                                 y * stride * scale : y * stride * scale + self.label_size, :]
                            input_seq.append(input_patch)
                            label_seq.append(label_patch)
                make_data(input_seq, label_seq, self.dataset,self.scale)
                print('Made {}th data group'.format(gp))
        else:
            input_seq, label_seq = [], []
            for data in self.data:
                input_, label_ = preprocess(data, scale)
                (height, width, _) = input_.shape
                x_range = (height - self.image_size) // stride
                y_range = (width - self.image_size) // stride
                for x in range(x_range):
                    for y in range(y_range):
                        input_patch = input_[x * stride : x * stride + self.image_size,
                                             y * stride : y * stride + self.image_size, :]
                        label_patch = label_[x * stride * scale : x * stride * scale + self.label_size,
                                             y * stride * scale : y * stride * scale + self.label_size, :]
                        input_seq.append(input_patch)
                        label_seq.append(label_patch)
            make_data(input_seq, label_seq, self.dataset, self.scale)
        print("Make dataset done...")


def prepare_data(dataset, image_form):
    """
    Args:
        dataset: the data directory, str
        image_forom: image file type, str
    """
    if not os.path.exists(dataset):
        raise Exception("No such dataset")
    data_files = glob.glob(os.path.join(dataset, '*_HR'+image_form))
    return data_files

def preprocess(path, scale):
    """
    Read image and make a pair of input and label
    """
    image = Image.open(path).convert('RGB')
    (width, height) = image.size
    label_ = np.array(list(image.getdata())).astype(np.float32).reshape((height, width, -1)) / 255

    new_height, new_width = int(height / scale), int(width / scale)
    scaled_img = image.resize((new_width, new_height), Image.ANTIALIAS)
    input_ = np.array(list(scaled_img.getdata())).astype(np.float32).reshape((new_height, new_width, -1)) / 255
    image.close()

    return input_, label_

def make_data(data, label, dataset, scale):
    """
    Input data, make h5 file for training
    """
    assert(len(label) == len(data))
    image_size = data[0].shape[0]
    label_size = label[0].shape[0]
    save_path = os.getcwd() + os.sep + 'cache' + os.sep + dataset + '_train_X{}.h5'.format(scale)  # ./RDN/cahce/Set5_train_X3.h5
    if not os.path.exists(save_path):
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset('data', data=data, chunks=True, maxshape=[None, image_size, image_size, 3])
            hf.create_dataset('label', data=label, chunks=True, maxshape=[None, label_size, label_size, 3])
    else:
        with h5py.File(save_path, 'a') as hf:
            len1 = hf['data'].shape[0]
            len2 = data.shape[0]
            hf['data'].resize((len1 + len2), axis=0)
            hf['data'][-len2:] = data
            len3 = hf['label'].shape[0]
            len4 = label.shape[0]
            hf['label'].resize((len3+len4), axis=0)
            hf['label'][-len4:] = label


if __name__ == "__main__":
    data = Data('data'+os.sep+'Set5', '.png', 2)
    data.train_setup()
        