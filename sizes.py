"""
https://github.com/tensorflow/models/tree/master/research/slim
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_sizes = {
    'mobilenet_v1_050': 160,
    'mobilenet_v1_025': 128,
    'mobilenet_v2_140': 224,
    'mobilenet_v2_035': 224,
    'mobilenet_v2': 224,
}


def get_trained_size(name):
    return _sizes.get(name, None)
