import os.path as osp
import sys
import json
import yaml
import copy
from addict import Dict
from importlib import import_module
from functools import wraps


__all__ = ['Config']


def check_frozen(func):
    @wraps(func)
    def decorated(self, *args, **kwargs):
        if self.is_frozen:
            raise AttributeError(
                'Attempt to modify frozen Config object.')
        else:
            return func(self, *args, **kwargs)
    return decorated


class Config(Dict):
    
    def __init__(self, *args, **kwargs):
        self.__dict__['__frozen__'] = False
        super(Config, self).__init__(*args, **kwargs)

    @classmethod
    def load(cls, filename, **kwargs):
        if not osp.exists(filename):
            raise FileNotFoundError(
                'No such file or directory: {}'.format(filename))
        name, ext = osp.splitext(filename)
        assert ext in ['.py', '.json', '.yaml', '.yml']
        if ext == '.py':
            if '.' in name:
                raise ValueError(
                    'Dots are not allowed in config file path.')
            # load config module
            cfg_dir = osp.dirname(filename)
            sys.path.insert(0, cfg_dir)
            module = import_module(name)
            sys.path.pop(0)
            # parse config dictionary
            cfg_dict = {k: v for k, v in module.__dict__.items()
                        if not k.startswith('__')}
        elif ext == '.json':
            with open(filename, 'r') as f:
                cfg_dict = json.load(f, **kwargs)
        elif ext in ['.yaml', '.yml']:
            with open(filename, 'r') as f:
                kwargs.setdefault('Loader', yaml.SafeLoader)
                cfg_dict = yaml.load(f, **kwargs)
        return Config(cfg_dict)
    
    def dump(self, filename, **kwargs):
        name, ext = osp.splitext(filename)
        assert ext in ['.py', '.json', '.yaml', '.yml']
        if ext == '.py':
            raise NotImplementedError(
                'Saving to .py files is not supported.')
        elif ext == '.json':
            kwargs.setdefault('indent', 4)
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f, **kwargs)
        elif ext in ['.yaml', '.yml']:
            with open(filename, 'w') as f:
                yaml.safe_dump(self.to_dict(), f, **kwargs)
    
    def merge_from(self, cfg):
        merged_cfg = self.deepcopy()
        merged_cfg.update(cfg)
        return merged_cfg
    
    def merge_to(self, cfg):
        merged_cfg = cfg.deepcopy()
        merged_cfg.update(self)
        return merged_cfg
    
    @check_frozen
    def __setattr__(self, name, value):
        super(Config, self).__setattr__(name, value)
    
    @check_frozen
    def __setitem__(self, name, value):
        super(Config, self).__setitem__(name, value)
    
    @check_frozen
    def __setstate__(self, state):
        super(Config, self).__setstate__(state)
    
    @check_frozen
    def setdefault(self, key, default=None):
        super(Config, self).setdefault(key, default)
    
    @check_frozen
    def __delattr__(self, name):
        super(Config, self).__delattr__(name)
    
    @check_frozen
    def update(self, *args, **kwargs):
        super(Config, self).update(*args, **kwargs)
    
    def deepcopy(self):
        cfg = copy.deepcopy(self)
        if cfg.is_frozen:
            cfg.defrost()
        return cfg

    def freeze(self):
        self.__dict__['__frozen__'] = True
        for v in self.values():
            if isinstance(v, Config):
                v.freeze()
    
    def defrost(self):
        self.__dict__['__frozen__'] = False
        for v in self.values():
            if isinstance(v, Config):
                v.defrost()
    
    @property
    def is_frozen(self):
        return self.__dict__.get('__frozen__', False)
