import abc


class Evaluator(object):
    __metaclass__  = abc.ABCMeta
    
    @abc.abstractmethod
    def run(self, model, visualize=False):
        raise NotImplementedError
    
    @abc.abstractmethod
    def report(self, model_names):
        raise NotImplementedError
    
    @abc.abstractmethod
    def show(self, model_names):
        raise NotImplementedError
