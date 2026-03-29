import os
import torch
import yaml
import random
import json
import numpy as np
import logging
import os.path as osp
import pickle
import shutil
import sys
import time
import warnings
from collections import defaultdict, OrderedDict
from functools import partial


def set_seed(seed):
    if seed == -1: seed = random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def setup_logger(output_dir):
    logger = logging.getLogger("FlowAdapter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def save_checkpoint(model, optimizer, scheduler, config, epoch, output_dir, is_best=False):
    # This explicitly saves the OP matrix (self.OP.W) because it is part of the model state_dict
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'config': config,
    }
    # Save independent OP matrix for easy analysis if needed
    if hasattr(model, 'OP') and model.OP.W is not None:
        state['OP'] = model.OP.W

    filename = os.path.join(output_dir, f"checkpoint.pth")
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(output_dir, "model_best.pth"))

def load_checkpoint(model, path, optimizer=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if 'OP' in checkpoint.keys():
        model.OP.W = checkpoint['OP'].float()

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"Loaded model from epoch {checkpoint.get('epoch', 0)}")
    return checkpoint.get('config', None)


class Registry:
    """A registry providing name -> object mapping, to support
    custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone(nn.Module):
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = dict()

    def _do_register(self, name, obj, force=False):
        if name in self._obj_map and not force:
            raise KeyError(
                'An object named "{}" was already '
                'registered in "{}" registry'.format(name, self._name)
            )

        self._obj_map[name] = obj

    def register(self, obj=None, force=False):
        if obj is None:
            # Used as a decorator
            def wrapper(fn_or_class):
                name = fn_or_class.__name__
                self._do_register(name, fn_or_class, force=force)
                return fn_or_class

            return wrapper

        # Used as a function call
        name = obj.__name__
        self._do_register(name, obj, force=force)

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError(
                'Object name "{}" does not exist '
                'in "{}" registry'.format(name, self._name)
            )

        return self._obj_map[name]

    def registered_names(self):
        return list(self._obj_map.keys())

def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile
def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items
def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x
def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj
def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))