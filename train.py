import os, glob, argparse, time, shutil

import numpy as np
import tensorflow as tf
import h5py
from PIL import Image

from rdn import rdn
from data import Data
import config as cfg
import skimage.color as sc
from calculate import calculate_metrics

class Trainer(object):
    def __init__(self, net, data, scale, args):
        self.net = net
        self.data = data
        self.scale = scale
        self.image_size = cfg.IMAGE_SIZE
        self.epochs = cfg.EPOCHS
        self.batch = cfg.BATCH
        self.hf = None

        # parameters for training
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(cfg.L_R/2, 
                            global_step=self.global_step, decay_steps=200000, decay_rate=0.5, staircase=True)
        # saver and some paths
        variables_to_save = tf.global_variables()
        variables_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(variables_to_save, max_to_keep=1)
        self.bestSaver = tf.train.Saver(variables_to_save, max_to_keep=1)
        self.cache_name = args.dataset + '_train_X{}.h5'.format(self.scale)
        self.cache_dir = cfg.CACHE_DIR
        self.event_dir = cfg.EVENT_DIR
        self.weight_file = os.path.join(cfg.WEIGHTS_DIR, args.weight_file)
        self.bestWeight = os.path.join(os.path.join(cfg.WEIGHTS_DIR, 'best'), 'rdn_best')
        self.cache_file = os.path.join(self.cache_dir, self.cache_name)

        # build rdn net and train operation
        self.net.build_net()
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients = self.optimizer.compute_gradients(self.net.loss, tf.trainable_variables())
        self.train_op = self.optimizer.minimize(self.net.loss, global_step=self.global_step)

        # add scalars which can be showed on TensorBoard
        with tf.name_scope('performance'):
            self.tf_loss_ph = tf.placeholder(tf.float32, [], name='loss_ph')
            self.tf_psnr_ph = tf.placeholder(tf.float32, [], name='psnr_ph')
            self.tf_ssim_ph = tf.placeholder(tf.float32, [], name='ssim_ph')
            tf_loss_summary = tf.summary.scalar('loss', self.tf_loss_ph)
            tf_psnr_summary = tf.summary.scalar('psnr', self.tf_psnr_ph)
            tf_ssim_summary = tf.summary.scalar('ssim', self.tf_ssim_ph)
        self.performance_summary = tf.summary.merge([tf_loss_summary, tf_psnr_summary, tf_ssim_summary])

        with tf.name_scope('grads_and_learning_rate'):
            last_grads = self.gradients[-2][0]  # last layer's weights
            self.last_grads_norm = tf.sqrt(tf.reduce_mean(last_grads**2))
            tf_lr_summary = tf.summary.scalar('learning_rate', self.learning_rate)
            tf_grads_summary = tf.summary.scalar('grads_norm', self.last_grads_norm)
        self.grads_and_lr = tf.summary.merge([tf_grads_summary, tf_lr_summary])

        # add other preparation
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.sess = tf.Session(config=config)
        self.writer = tf.summary.FileWriter(self.event_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        if not args.fresh:
            self.restorer = tf.train.Saver(variables_to_restore, max_to_keep=1)
            self.restorer.restore(self.sess, self.weight_file)

    def train(self):
        # param for training
        epochs = self.epochs
        batch  = self.batch
        overall_time = time.time()
        if not os.path.exists(self.cache_file):
            self.data.train_setup()
        # training data
        self.hf = h5py.File(self.cache_file, 'r')
        data_len = len(self.hf['data'])
        counter = 0
        best_psnr = 36.35
        assert(len(self.hf['data']) == len(self.hf['label']))
        print("Run [1000] steps per epoch...")
        # testing data
        data_dir = os.path.join('data', os.path.join('Set5', 'image_SRF_{}'.format(scale)))
        lr_list = glob.glob(os.path.join(data_dir, '*_LR.png'))
        hr_list = glob.glob(os.path.join(data_dir, '*_HR.png'))
        # start training
        for epo in range(epochs):
            for step in range(1000):
                # random choose a batch
                rand_idx = np.random.randint(data_len - batch - 1)
                input_, label_ = self.hf['data'][rand_idx : rand_idx + batch], \
                                 self.hf['label'][rand_idx : rand_idx + batch]
                # add data augumentation
                random_aug = np.random.rand(2)
                input_, label_ = self.data.augument(random_aug, input_, label_)
                feed_dict = {self.net.input_: input_, self.net.label_: label_, self.net.batch:batch}
                _, err = self.sess.run([self.train_op, self.net.loss], feed_dict=feed_dict)
                counter += 1
                # print every 100 steps
                if counter % 100 == 0:
                    print("Training step: [{:6}], time: [{:4.6f}min], loss: [{:.6f}]".format(\
                        counter, (time.time()-overall_time)/60, err))
                # test on Set5 every 200 steps
                if counter % 200 == 0:
                    cur_psnr = self.test_on_Set5(err, counter, lr_list, hr_list)
                    if cur_psnr > best_psnr:
                        best_psnr = cur_psnr
                        self.bestSaver.save(self.sess, self.bestWeight)
                # save ckpt every 500 steps:
                if counter % 500 == 0:
                    gn_summ = self.sess.run(self.grads_and_lr, feed_dict={self.net.input_: input_, self.net.label_: label_})
                    self.writer.add_summary(gn_summ, counter)
                    self.saver.save(self.sess, self.weight_file, global_step=self.global_step)
            print("===================Epoch: [{:3}]===================".format(epo+1))
        print("Done...")

    def test_on_Set5(self, loss, step, lr_list, hr_list):
        scale = self.scale
        assert(len(lr_list)==len(hr_list))

        metrics = []
        for idx in range(len(hr_list)):
            # read pairs
            lr_img = Image.open(lr_list[idx])
            hr_img = Image.open(hr_list[idx])
            width, height = lr_img.size
            width_hr, height_hr = width * scale, height * scale
            hr_img = hr_img.resize((width_hr, height_hr), Image.ANTIALIAS)
            # convert to NumPy
            lr_image = np.array(list(lr_img.getdata())).astype(np.float32).reshape([height, width, -1]) / 255.0
            hr_image = np.array(list(hr_img.getdata())).astype(np.uint8).reshape([height_hr, width_hr, -1])
            lr_image = lr_image[np.newaxis, :] # NHWC
            feed_dict = {self.net.input_: lr_image}
            output = self.sess.run(self.net.output, feed_dict=feed_dict)[0]
            output = (output * 255).astype(np.uint8)

            label_ycbcr = sc.rgb2ycbcr(hr_img)
            output_ycbcr = sc.rgb2ycbcr(output)
            metrics.append(calculate_metrics([label_ycbcr[:, :, 0:1]], [output_ycbcr[:, :, 0:1]]))
            lr_img.close()
            hr_img.close()
        avg_psnr = sum([m[0] for m in metrics]) / len(metrics)
        avg_ssim = sum([m[1] for m in metrics]) / len(metrics)
        feed_dict={self.tf_loss_ph: loss, self.tf_psnr_ph: avg_psnr, self.tf_ssim_ph: avg_ssim}
        summ = self.sess.run(self.performance_summary, feed_dict=feed_dict)
        self.writer.add_summary(summ, step)
        return avg_psnr

    def __del__(self):
        if self.sess:
            self.sess.close()
        if self.hf:
            self.hf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DIV2K_train_HR', type=str)
    parser.add_argument('--image_form', default='.png', type=str)
    parser.add_argument('--fresh', default=False, type=bool)
    parser.add_argument('--scale', default=0, type=int)
    parser.add_argument('--weight_file', default='rdn.ckpt', type=str)
    args = parser.parse_args()

    # check status
    if args.scale == 0:
        raise Exception('Wrong scale\nUse --scale to specify the scale')
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
    
    scale = args.scale
    net = rdn(True, scale)
    data = Data(args.dataset, args.image_form, scale)

    recover = Trainer(net, data, scale, args)
    print("Start training...")
    recover.train()