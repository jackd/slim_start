#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from slim_start import get_starter
from slim_start import get_names
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, 'model name')
flags.DEFINE_bool(
    'continue_on_error', True,
    'Whether or not to continue once an error has been thrown.')


def _main(name):
    import tensorflow as tf
    starter = get_starter(name)
    continue_on_error = FLAGS.continue_on_error
    if starter.has_checkpoint:
        try:
            graph = tf.Graph()
            with graph.as_default():
                image = tf.placeholder(
                    shape=(None, 311, 311, 3), dtype=tf.float32)
                fn = starter.get_network_fn(is_training=True)
                fn(image)
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                saver = tf.train.Saver(var_list=var_list)

            with tf.Session(graph=graph) as sess:
                ckpt = starter.get_checkpoint()
                saver.restore(sess, ckpt)
                return True
        except Exception:
            if continue_on_error:
                return False
            else:
                raise
    else:
        return None


def main(_):
    name = FLAGS.name

    def state_str(state):
        return 'N/A' if state is None else 'success' if state else 'failed'

    if name is None:
        out = []
        names = tuple(get_names())
        for name in names:
            out.append(_main(name))
        for name, state in zip(names, out):
            print('%s: %s' % (name, state_str(state)))
    else:
        print('%s: %s' % (name, state_str(_main(name))))


if __name__ == '__main__':
    app.run(main)
