from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_default_weight_decay(name):
    vals = (
        ('alexnet_v2', 0.0005),
        ('cifarnet', 0.004),
        ('cyclegan', 0.0),
        ('inception_resnet', 0.00004),
        ('inception', 0.00004),
        ('lenet', 0.0),
        ('mobilenet_v1', 0.00004),
        ('mobilenet', 0.00004),
        ('nasnet_cifar', 5e-4),
        ('nasnet_mobile', 4e-5),
        ('nasnet_large', 5e-5),
        ('pnasnet_large', 4e-5),
        ('pnasnet_mobile', 4e-5),
        ('overfeat', 0.0005),
        ('resnet', 0.0001),
        ('vgg', 0.0005)
    )
    for start, val in vals:
        if name.startswith(start):
            return val
    raise KeyError('Unrecognized name "%s"' % name)
