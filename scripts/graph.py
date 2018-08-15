#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, 'model name')
flags.DEFINE_string(
    'collection', 'variables', 'collection name to print')
flags.DEFINE_string('scope', None, 'variable scope')
flags.DEFINE_boolean('endpoints', False, 'if true, prints endpoints')


def main(_):
    import tensorflow as tf
    from slim_start import get_starter
    name = FLAGS.name
    starter = get_starter(name)
    image = tf.zeros((2, 224, 224, 3), dtype=tf.float32)
    with tf.contrib.slim.arg_scope(starter.get_scope()):
        out, endpoints = starter.get_unscoped_network_fn(
            num_classes=None)(image)

    if FLAGS.endpoints:
        for k, v in endpoints.items():
            print('%s: %s' % (k, str(v.shape)))
    else:
        vars = tf.get_collection(FLAGS.collection, scope=FLAGS.scope)
        for var in vars:
            print(var.name)


if __name__ == '__main__':
    app.run(main)
