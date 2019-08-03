import os, glob, argparse, time, shutil

import numpy as np
import tensorflow as tf
import h5py

from rdn import rdn
from data import Data
import config as cfg

class Recover(object):
    def __init__(self, net, data, scale, args):
        self.net = net
        self.data = data
        self.scale = scale
        self.image_size = cfg.IMAGE_SIZE
        self.epochs = cfg.EPOCHS
        self.batch = cfg.BATCH

        self.learning_rate = cfg.L_R
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        variables_to_save = tf.global_variables()
        variables_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(variables_to_save, max_to_keep=2)
        self.cache_name = args.dataset + '_train_X{}.h5'.format(self.scale)
        self.cache_dir = cfg.CACHE_DIR
        self.event_dir = cfg.EVENT_DIR
        self.weight_file = os.path.join(cfg.WEIGHTS_DIR, args.weight_file)
        self.cache_file = os.path.join(self.cache_dir, self.cache_name)

        # build rdn net
        self.net.build_net([self.image_size, self.image_size])
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.net.loss, global_step=self.global_step)

        self.epochs = cfg.EPOCHS
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if not args.fresh:
            self.restorer = tf.train.Saver(variables_to_restore, max_to_keep=2)
            self.restorer.restore(self.sess, self.weight_file)

    def train(self):
        epochs = self.epochs
        batch  = self.batch
        overall_time = time.time()
        if not os.path.exists(self.cache_file):
            self.data.train_setup()

        hf = h5py.File(self.cache_file, 'r')
        data_len = len(hf['data'])
        assert(len(hf['data']) == len(hf['label']))
        steps = data_len // batch

        for epo in range(epochs):
            for step in range(1, steps+1):
                input_, label_ = hf['data'][step*batch : (step+1)*batch], hf['label'][step*batch : (step+1)*batch]
                feed_dict = {self.net.input_: input_, self.net.label_: label_, self.net.batch:batch}
                _, err = self.sess.run([self.train_op, self.net.loss], feed_dict=feed_dict)
                # print every 100 steps
                if step % 20 == 0:
                    print("Training step: [{}], time: [{}min], loss: [{}]".format(\
                        step+epo*steps, (time.time()-overall_time)/60, err))
                # save ckpt every 500 steps:
                if step % 40 == 0:
                    self.saver.save(self.sess, self.weight_file, global_step=self.global_step)
        print("Done...")
        hf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Set14', type=str)
    parser.add_argument('--image_form', default='.png', type=str)
    parser.add_argument('--fresh', default=True, type=bool)
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

    recover = Recover(net, data, scale, args)
    print("Start training...\n")
    recover.train()