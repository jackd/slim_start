#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, 'model name')
flags.DEFINE_string(
    'collection', 'global_variables', 'collection name to print')
flags.DEFINE_string('scope', None, 'variable scope')


def main(_):
    import tensorflow as tf
    from slim_start import get_starter
    name = FLAGS.name
    starter = get_starter(name)
    image = tf.zeros((2, 224, 224, 3), dtype=tf.float32)
    starter.get_network_fn(is_training=True)(image)

    vars = tf.get_collection(FLAGS.collection, scope=FLAGS.scope)
    for var in vars:
        print(var.name)


if __name__ == '__main__':
    app.run(main)
