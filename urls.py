"""
https://github.com/tensorflow/models/tree/master/research/slim
"""
# flake8: noqa
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_urls = {
    'alexnet_v2': None,
    'cifarnet': None,
    'overfeat': None,
    'vgg_a': None,
    'vgg_16': 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
    'vgg_19': 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
    'inception_v1': 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
    'inception_v2': 'http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz',
    'inception_v3': 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
    'inception_v4': 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
    'inception_resnet_v2': 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz',
    'lenet': None,
    'resnet_v1_50': 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
    'resnet_v1_101': 'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
    'resnet_v1_152': 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
    'resnet_v1_200': None,
    'resnet_v2_50': 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
    'resnet_v2_101': 'http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
    'resnet_v2_152': 'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
    'resnet_v2_200': None,
    'mobilenet_v1': 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz',
    'mobilenet_v1_075': None,
    'mobilenet_v1_050': 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz',
    'mobilenet_v1_025': 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz',
    'mobilenet_v2': 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz',
    'mobilenet_v2_140': 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz',
    'mobilenet_v2_035': None,
    'nasnet_cifar': None,
    'nasnet_mobile': 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz',
    'nasnet_large': 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz',
    'pnasnet_large': 'https://storage.googleapis.com/download.tensorflow.org/models/pnasnet-5_large_2017_12_13.tar.gz',
    'pnasnet_mobile': 'https://storage.googleapis.com/download.tensorflow.org/models/pnasnet-5_mobile_2017_12_13.tar.gz',
}

def get_names():
    return tuple(_urls.keys())


def get_url(name):
    return _urls.get(name, None)
