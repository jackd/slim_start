#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, 'model name')


def main(_):
    from tensorflow.python.tools.inspect_checkpoint import \
        print_tensors_in_checkpoint_file
    from slim_start import get_starter
    name = FLAGS.name
    starter = get_starter(name)
    latest_ckp = starter.get_checkpoint()
    print_tensors_in_checkpoint_file(
        latest_ckp, tensor_name='', all_tensors=False, all_tensor_names=True)


if __name__ == '__main__':
    app.run(main)
