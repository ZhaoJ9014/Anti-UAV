import jittor.nn as nn
import abc


class Model(nn.Module):
    __metaclass__  = abc.ABCMeta

    @abc.abstractmethod
    def forward_train(self, train_batch):
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward_val(self, val_batch):
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward_test(self, test_data, visualize=False):
        raise NotImplementedError
