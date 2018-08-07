from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six.moves.urllib as urllib


_checkpoints_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                '_checkpoints')

if not os.path.isdir(_checkpoints_dir):
    os.makedirs(_checkpoints_dir)


class Starter(object):
    def __init__(self, name, url):
        self._name = name
        self._url = url
        if url is None:
            self._model_id = None
        else:
            model_id = os.path.split(url)[-1]
            for s in ('.tar.gz', '.tgz'):
                if model_id.endswith(s):
                    model_id = model_id[:-len(s)]
                    break
            else:
                raise ValueError('Invalid url "%s"' % url)
            self._model_id = model_id

    @property
    def name(self):
        return self._name

    @property
    def url(self):
        return self._url

    @property
    def model_id(self):
        return self._model_id

    def get_network_fn(
            self, num_classes=None, weight_decay=0.0, is_training=False):
        from slim.nets.nets_factory import get_network_fn
        return get_network_fn(
            self.name, num_classes, weight_decay, is_training)

    def default_image_size(self):
        from .sizes import get_trained_size
        size = get_trained_size(self.name)
        if size is None:
            fn = self.get_network_fn()
            if hasattr(fn, 'default_image_size'):
                size = fn.default_image_size
        return size

    def clean(self):
        import shutil
        f = self.checkpoint_dir
        if os.path.isdir(f):
            shutil.rmtree(f)
        self.clean_archive()

    def clean_archive(self):
        path = self.download_path
        if os.path.isfile(path):
            os.remove(path)
            return True
        else:
            return False

    @property
    def download_path(self):
        filename = os.path.split(self.url)[1]
        return os.path.join(_checkpoints_dir, filename)

    @property
    def checkpoint_dir(self):
        model_id = self.model_id
        if model_id is None:
            raise ValueError('No checkpoint_dir for model "%s"' % self.name)
        else:
            return os.path.join(_checkpoints_dir, model_id)

    def _extract(self):
        import tarfile
        url = self.url
        if url is None:
            return None
        path = self.download_path
        if not os.path.isfile(path):
            print('Downloading from %s ...' % url)
            try:
                opener = urllib.request.URLopener()
                opener.retrieve(url, path)
            except IOError:
                print(
                    'Problem downloading starter file from "%s" for model '
                    '"%s".' % (url, self.name))
                raise
            print('Done!')

        print('Extracting...')
        with tarfile.open(path) as tar_file:
            tar_file.extractall(self.checkpoint_dir)
        print('Done!')

    def get_checkpoint(self):
        if self.has_checkpoint:
            checkpoint_dir = self.checkpoint_dir
            if not os.path.isdir(checkpoint_dir):
                try:
                    self._extract()
                except IOError:
                    return None
            assert(os.path.isdir(checkpoint_dir))
            path = os.path.join(checkpoint_dir, '%s.ckpt' % self.name)
            if os.path.isfile(path):
                return path
            else:
                path = os.path.join(checkpoint_dir, '%s.ckpt' % self.model_id)
                if os.path.isfile('%s.meta' % path):
                    return path
                else:
                    fns = os.path.listdir(checkpoint_dir)
                    return RuntimeError(
                        'Cannot find checkpoint file. '
                        'Directory contents: %s' % str(fns))

        else:
            raise RuntimeError(
                'No checkpoint available for model "%s"' % self.name)

    @property
    def has_checkpoint(self):
        return self.model_id is not None


def get_starter(name):
    from .urls import get_url
    return Starter(name, get_url(name))
