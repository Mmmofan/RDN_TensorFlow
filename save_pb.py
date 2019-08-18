import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util

from rdn import rdn

def make_pb(scale, pb_file):
    rdn = rdn(False, scale)
    rdn.build_net()
    variables_to_restore = tf.global_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt_path = 'weights'
    ckpt_file = os.path.join(ckpt_path, 'rdn.ckpt')

    init = tf.global_variables_initializer()
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['input', 'add_168'])
    with tf.gfile.FastGFile(pb_file, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

if __name__ == "__main__":
    make_pb(2, 'rdn_pb.pb')