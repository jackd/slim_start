#!/usr/bin/python
"""
Example usage of `slim_start` training MobileNetV2 for MNIST.

Obviously the model is overkill for MNIST... but it serves its purpose.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from slim_start import get_starter
try:
    import official.mnist.dataset as ds
except ImportError:
    print(
        'No `official` found on your path.\n'
        'Please clone tensorflow/models and add to your `PYTHONPATH`, e.g.\n'
        'cd ~\n'
        'git clone https://github.com/tensorflow/models.git\n'
        'export PYTHONPATH=$PYTHONPATH:~/models')
    raise
slim = tf.contrib.slim
ModeKeys = tf.estimator.ModeKeys


def get_inputs(mode, batch_size=64):
    """
    Get batched (features, labels) from mnist.

    Args:
        `mode`: string representing mode of inputs.
            Should be one of {"train", "eval", "predict", "infer"}

    Returns:
        `features`: float32 tensor of shape (batch_size, 28, 28, 1) with
            grayscale values between 0 and 1.
        `labels`: int32 tensor of shape (batch_size,) with labels indicating
            the digit shown in `features`.
    """
    # Get the base dataset
    if mode == ModeKeys.TRAIN:
        dataset = ds.train('/tmp/mnist_data')
    elif mode in {ModeKeys.PREDICT, ModeKeys.EVAL}:
        dataset = ds.test('/tmp/mnist_data')
    else:
        raise ValueError(
            'mode must be one in ModeKeys')

    # repeat and shuffle if training
    if mode == 'train':
        dataset = dataset.repeat()  # repeat indefinitely
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch_size)

    image, labels = dataset.make_one_shot_iterator().get_next()
    image = tf.cast(tf.reshape(image, (-1, 28, 28, 1)), tf.float32)
    return image, labels


def get_spec(features, labels, mode, logits_fn):
    features = tf.tile(features, (1, 1, 1, 3))
    logits = logits_fn(features, mode=mode)
    predictions = tf.argmax(logits, axis=-1)
    kwargs = dict(mode=mode, predictions=predictions)
    if mode == ModeKeys.PREDICT:
        pass
    else:
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)

        classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        classification_loss = tf.reduce_sum(classification_loss)
        tf.losses.add_loss(classification_loss)
        loss = slim.losses.get_total_loss()

        optimizer = tf.train.AdamOptimizer()
        train_op = slim.learning.create_train_op(loss, optimizer)
        kwargs['loss'] = loss
        kwargs['eval_metric_ops'] = dict(accuracy=accuracy)
        kwargs['train_op'] = train_op
    return tf.estimator.EstimatorSpec(**kwargs)


name = 'mobilenet_v2'
weight_decay = 0.0
num_classes = 10
bn_decay = 0.9

starter = get_starter(name)
vars_to_warm_start = 'MobilenetV2/*'
warm_start_settings = tf.estimator.WarmStartSettings(
    starter.get_checkpoint(), vars_to_warm_start)


def logits_fn(features, mode):
    is_training = mode == ModeKeys.TRAIN
    f = starter.get_scoped_network_fn(
        is_training=is_training, weight_decay=weight_decay, bn_decay=bn_decay)
    x, _ = f(features, base_only=True)
    x = tf.reduce_mean(x, axis=(1, 2))
    x = tf.layers.dense(x, num_classes)
    return x


models_dir = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)


estimator = tf.estimator.Estimator(
    lambda features, labels, mode: get_spec(features, labels, mode, logits_fn),
    model_dir=os.path.join(models_dir, '%s-mnist' % name),
    warm_start_from=warm_start_settings)

tf.logging.set_verbosity(tf.logging.INFO)
estimator.train(lambda: get_inputs(ModeKeys.TRAIN), max_steps=10000)
e0 = estimator.evaluate(lambda: get_inputs(ModeKeys.EVAL))
print(e0)
