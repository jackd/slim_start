#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from slim_start import get_starter
from slim_start import get_names
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, 'model name')


def _main(name):
    starter = get_starter(name)
    print(name)
    if starter.url is not None:
        if starter.clean_archive():
            print('Cleaned')
        else:
            print('No archive present')
    else:
        print('No url')
    print('---------------------------------------')


def main(_):
    name = FLAGS.name
    if name is None:
        for name in get_names():
            _main(name)
    else:
        _main(name)


if __name__ == '__main__':
    app.run(main)
