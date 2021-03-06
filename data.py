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
        self.data = prepare_data(os.path.join(self.data_dir, self.dataset), self.image_form)
        if len(self.data) >= 100:  # in case the data is too large
            group = len(self.data) // 50
            for gp in range(group):
                data_list = self.data[gp * 50 : (gp+1) * 50]
                input_seq, label_seq = [], []
                for data in data_list:
                    input_, label_ = self.preprocess(data, scale)
                    (height, width, _) = input_.shape
                    x_range = (height - self.image_size) // stride
                    y_range = (width - self.image_size) // stride
                    for x in range(x_range):
                        for y in range(y_range):
                            input_patch = input_[x * stride : x * stride + self.image_size,
                                                 y * stride : y * stride + self.image_size, :]
                            label_patch = label_[x * stride * scale : x * stride * scale + self.label_size,
                                                 y * stride * scale : y * stride * scale + self.label_size, :]
                            # make sure the patch has enough edge for good training 
                            # reference: https://github.com/hengchuan/RDN-TensorFlow/blob/master/utils.py
                            t = cv2.cvtColor(label_patch, cv2.COLOR_BGR2YCR_CB)
                            t = t[:, :, 0] # y
                            gx = (t[1:, 0:-1] - t[0:-1, 0:-1]) * 255
                            gy = (t[0:-1, 1:] - t[0:-1, 0:-1]) * 255
                            Gxy = (gx**2 + gy**2)**0.5
                            r_gxy = float((Gxy>10).sum()) / ((self.image_size*scale)**2) * 100
                            if r_gxy < 10:
                                continue

                            input_seq.append(input_patch)
                            label_seq.append(label_patch)
                make_data(input_seq, label_seq, self.dataset,self.scale)
                print('Made {}th/{} data group'.format(gp+1, group))
        else:
            input_seq, label_seq = [], []
            for data in self.data:
                input_, label_ = self.preprocess(data, scale)
                (height, width, _) = input_.shape
                x_range = (height - self.image_size) // stride
                y_range = (width - self.image_size) // stride
                for x in range(x_range):
                    for y in range(y_range):
                        input_patch = input_[x * stride : x * stride + self.image_size,
                                             y * stride : y * stride + self.image_size, :]
                        label_patch = label_[x * stride * scale : x * stride * scale + self.label_size,
                                             y * stride * scale : y * stride * scale + self.label_size, :]
                        t = cv2.cvtColor(label_patch, cv2.COLOR_BGR2YCR_CB)
                        t = t[:, :, 0] # y
                        gx = t[1:, 0:-1] - t[0:-1, 0:-1]
                        gy = t[0:-1, 1:] - t[0:-1, 0:-1]
                        Gxy = (gx**2 + gy**2)**0.5
                        r_gxy = float((Gxy>10).sum()) / ((self.image_size*scale)**2) * 100
                        if r_gxy < 10:
                            continue

                        input_seq.append(input_patch)
                        label_seq.append(label_patch)
            make_data(input_seq, label_seq, self.dataset, self.scale)
        print("Make dataset done...")

    def augument(self, random, input_, label_):
        """
        random: [rand1, rand2]
        """
        if random[0] < 0.3:
            input_ = np.flip(input_, 1)
            label_ = np.flip(label_, 1)
        elif random[0] > 0.7:
            input_ = np.flip(input_, 2)
            label_ = np.flip(label_, 2)
        else:
            pass

        if random[1] < 0.5:
            input_ = np.rot90(input_, 1, [1, 2])
            label_ = np.rot90(label_, 1, [1, 2])

        return input_, label_

    def preprocess(self, path, scale):
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

def prepare_data(dataset, image_form):
    """
    Args:
        dataset: the data directory, str
        image_forom: image file type, str
    """
    if not os.path.exists(dataset):
        raise Exception("No such dataset")
    data_files = glob.glob(os.path.join(dataset, '*'+image_form))
    return data_files

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
            len2 = len(data)
            hf['data'].resize((len1 + len2), axis=0)
            hf['data'][-len2:] = data
            len3 = hf['label'].shape[0]
            len4 = len(label)
            hf['label'].resize((len3+len4), axis=0)
            hf['label'][-len4:] = label


if __name__ == "__main__":
    data = Data('data'+os.sep+'Set5', '.png', 2)
    data.train_setup()
        