import os, glob, argparse, time
import numpy as np
import tensorflow as tf
from rdn import rdn
from data import Data
from calculate import calculate_metrics
import cv2
from PIL import Image
import skimage.color as sc
import config as cfg

class Restorer(object):
    def __init__(self, net, data, scale, args):
        self.net = net
        self.data = data
        self.args = args
        self.net.build_net()
        
        self.scale = scale
        self.weights_dir = cfg.WEIGHTS_DIR
        self.weight_file = os.path.join(self.weights_dir, args.weight_file)
        variables_to_restore = tf.global_variables()
        self.restorer = tf.train.Saver(variables_to_restore)

        self.data_dir  = cfg.DATA_DIR
        self.data_file_path = os.path.join(self.data_dir, args.dataset)
        self.out_dir = cfg.OUT_DIR

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.sess = tf.Session(config=config)
        self.restorer.restore(self.sess, self.weight_file)

    def valid(self):
        scale = self.scale
        data_files = glob.glob(os.path.join(self.data_file_path+ os.sep + 'image_SRF_{}'.format(scale), '*_HR{}'.format(self.args.image_form)))

        metrics_1, metrics_2 = [], []
        for data in range(len(data_files)):
            img = Image.open(data_files[data])
            if img.mode != 'RGB':
                continue
            (width, height) = img.size
            (width_in, height_in) = width // scale, height // scale
            (width_lb, height_lb) = width_in * scale, height_in * scale
            label_ = img.resize((width_lb, height_lb), Image.ANTIALIAS)  # as label
            input_ = img.resize((width_in, height_in), Image.ANTIALIAS)  # as input 
            valid_ = input_.resize((width_lb, height_lb), Image.ANTIALIAS)  # as bicubic

            label_ = np.array(list(label_.getdata())).astype(np.float32).reshape([height_lb, width_lb, -1]) / 255
            input_ = np.array(list(input_.getdata())).astype(np.float32).reshape([height_in, width_in, -1]) / 255
            valid_ = np.array(list(valid_.getdata())).astype(np.float32).reshape([height_lb, width_lb, -1]) / 255

            feed_input = input_[np.newaxis, :]
            click = time.time()
            feed_dict = {self.net.input_: feed_input}
            output = self.sess.run(self.net.output, feed_dict=feed_dict)[0]
            print('Process image with shape: {:d} x {:d}, take time: {:.3f}s'.format(height_in, width_in, time.time()-click))
            if self.args.save:
                array_image_save(output * 255, os.path.join(self.out_dir, '{}_{}.png'.format(self.args.dataset, data)))
            output = np.clip((output * 255), 0, 255).astype(np.uint8)
            valid_ = np.clip((valid_ * 255), 0, 255).astype(np.uint8)
            label_ = np.clip((label_ * 255), 0, 255).astype(np.uint8)

            output_ycbcr = sc.rgb2ycbcr(output)
            hr_ycbcr = sc.rgb2ycbcr(label_)
            valid_ycbcr = sc.rgb2ycbcr(valid_)
            metrics_1.append(calculate_metrics([output_ycbcr[:, :, 0:1]], [hr_ycbcr[:, :, 0:1]]))
            metrics_2.append(calculate_metrics([valid_ycbcr[:, :, 0:1]], [hr_ycbcr[:, :, 0:1]]))
            img.close()
            avg_psnr1 = sum(m[0] for m in metrics_1) / len(metrics_1)
            avg_ssim1 = sum(m[1] for m in metrics_1) / len(metrics_1)
            avg_psnr2 = sum(m[0] for m in metrics_2) / len(metrics_2)
            avg_ssim2 = sum(m[1] for m in metrics_2) / len(metrics_2)
        return [avg_psnr1, avg_ssim1], [avg_psnr2, avg_ssim2]


def array_image_save(image, path):
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(path)
    print('Image saved: {}'.format(path))


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
    restorer = Restorer(net, data, scale, args)
    valid, gt = restorer.valid()
    print('Avg test PSNR:\t{:.4f}, \nAvg test SSIM:\t{:.4f}'.format(valid[0], valid[1]))
    print('Avg  BI  PSNR:\t{:.4f}, \nAvg  BI  SSIM:\t{:.4f}'.format(gt[0], gt[1]))