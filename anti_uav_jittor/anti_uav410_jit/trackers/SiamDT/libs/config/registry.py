import inspect
import six
import jittor as jt
import jittor.nn as nn
import jittor.optim as optim
# import torch.utils.data as data
import addict

import libs.ops as ops


__all__ = ['Registry', 'registry']


class Registry(object):

    def __init__(self):
        self._module_dict = {}
        self._init_python_modules()
        self._init_pytorch_modules()
    
    def _init_python_modules(self):
        # register built-in types
        modules = [tuple, list, dict, set]
        for m in modules:
            self.register_module(m, prefix='python')
    
    def _init_pytorch_modules(self):
        prefix = 'torch'

        # register all nn modules
        for k, v in nn.__dict__.items():
            if not isinstance(v, type):
                continue
            if issubclass(v, nn.Module) and \
                v is not nn.Module:
                self.register_module(v, prefix)

        # register all optimizers
        for k, v in optim.__dict__.items():
            if not isinstance(v, type):
                continue
            if issubclass(v, optim.Optimizer) and \
                v is not optim.Optimizer:
                self.register_module(v, prefix)
        
        # register all lr_schedulers
        for k, v in optim.LRScheduler.__dict__.items():
            if not isinstance(v, type):
                continue
            if issubclass(v, optim.LRScheduler) and \
                v is not optim.LRScheduler:
                self.register_module(v, prefix)
        
        # register all samplers
        for k, v in jt.dataset.Sampler.__dict__.items():
            if not isinstance(v, type):
                continue
            if issubclass(v, jt.dataset.Sampler) and \
                v is not jt.dataset.Sampler:
                self.register_module(v, prefix)
        
        # register datasets and dataloader
        for k, v in jt.dataset.__dict__.items():
            if not isinstance(v, type):
                continue
            if issubclass(v, jt.dataset.Dataset) and \
                v is not jt.dataset.Dataset:
                self.register_module(v, prefix)
        self.register_module(jt.dataset.DataLoader, prefix)
    
    def register_module(self, module, prefix=''):
        if not inspect.isclass(module) and \
            not inspect.isfunction(module):
            raise TypeError(
                'module must be a class or a function, '
                'but got {}'.format(module))
        module_name = module.__name__
        if prefix != '':
            module_name = '{}.{}'.format(prefix, module_name)
        if module_name in self._module_dict:
            raise KeyError('{} is already registered'.format(
                module_name))
        self._module_dict[module_name] = module
        return module

    def get(self, name):
        return self._module_dict.get(name, None)

    def build(self, cfg):
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg = cfg.copy()
        module_name = cfg.pop('type')
        if 'input_type' in cfg:
            ops.sys_print('Warning: "input_type" should be parsed '
                          'before building module')
            cfg.pop('input_type')

        # get module
        if isinstance(module_name, six.string_types):
            module = self.get(module_name)
            if module.__name__ == 'dict':
                module = addict.Dict
            if module is None:
                raise KeyError(
                    '{} is not in the registry'.format(module_name))
        else:
            raise TypeError(
                'type must be a string, but got {}'.format(module_name))
        
        # build submodules
        for k, v in cfg.items():
            if isinstance(v, dict) and 'type' in v:
                cfg[k] = self.build(v)

        return module(**cfg)
    
    @property
    def module_dict(self):
        return self._module_dict
    
    def __repr__(self):
        repr_str = self.__class__.__name + '(items={})'.format(
            list(self._module_dict.keys()))
        return repr_str


registry = Registry()
