import os
import os.path as osp
import pickle
import six
import abc
# from torch.utils.data import Dataset
from jittor.dataset import Dataset
from libs import ops


__all__ = ['SeqDataset', 'ImageDataset', 'PairDataset',
           'InstanceDataset']


class SeqDataset(Dataset):
    CLASSES = ('object', )

    def __init__(self, name, cache_dir='cache', **kwargs):
        self.name = name
        self.cache_dir = cache_dir
        self.cache_file = osp.join(cache_dir, name + '.pkl')

        # load or construct 'seq_dict'
        if not osp.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if osp.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                seq_dict = pickle.load(f)
        else:
            seq_dict = self._construct_seq_dict(**kwargs)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(seq_dict, f)

        # store seq_dict and extract seq_names
        self.seq_dict = seq_dict
        self.seq_names = list(seq_dict.keys())
    
    def __getitem__(self, index):
        r"""
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            img_files, target (tuple): where ``img_files`` is a list of
                file names and ``target`` is dict of annotations.
        """
        if isinstance(index, six.string_types):
            seq_name = index
        else:
            seq_name = self.seq_names[index]
        seq = self.seq_dict[seq_name]

        # parse img_files and target
        img_files, target = seq['img_files'], seq['target']
        if hasattr(self, 'transforms') and self.transforms is not None:
            img_files, target = self.transforms(img_files, target)
        
        return img_files, target

    def __len__(self):
        return len(self.seq_dict)
    
    @abc.abstractmethod
    def _construct_seq_dict(self, **kwargs):
        raise NotImplementedError


class ImageDataset(Dataset):
    CLASSES = ('object', )
    
    def __init__(self, name, cache_dir='cache', **kwargs):
        self.name = name
        self.cache_dir = cache_dir
        self.cache_file = osp.join(cache_dir, name + '.pkl')

        # load or construct 'img_dict'
        if not osp.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if osp.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                img_dict = pickle.load(f)
        else:
            img_dict = self._construct_img_dict(**kwargs)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(img_dict, f)

        # store img_dict and extract img_names
        self.img_dict = img_dict
        self.img_names = list(img_dict.keys())
    
    def __getitem__(self, index):
        r"""
        Args:
            index (integer): Index of an image.
        
        Returns:
            img, target (tuple): where ``target`` is dict of annotations.
        """
        img_name = self.img_names[index]
        img_info = self.img_dict[img_name]

        # parse image and annotations
        img = ops.read_image(img_info['img_file'])
        target = img_info['target']
        if hasattr(self, 'transforms') and self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.img_dict)
    
    @abc.abstractmethod
    def _construct_img_dict(self, **kwargs):
        raise NotImplementedError


class PairDataset(Dataset):
    CLASSES = ('object', )
    
    def __init__(self, name):
        self.name = name
    
    def __getitem__(self, index):
        r"""
        Args:
            index (integer): Index of an image.
        
        Returns:
            img_z, img_x, target (tuple): where ``target`` 
                is a dict of annotations.
        """
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError


class InstanceDataset(Dataset):

    def __init__(self, name, cache_dir='cache', **kwargs):
        self.name = name
        self.cache_dir = cache_dir
        self.cache_file = osp.join(cache_dir, name + '.pkl')

        # load or construct 'ins_dict'
        if not osp.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if osp.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                ins_dict = pickle.load(f)
        else:
            ins_dict = self._construct_ins_dict(**kwargs)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(ins_dict, f)

        # store ins_dict and extract ins_names
        self.ins_dict = ins_dict
        self.ins_names = list(ins_dict.keys())
    
    def __getitem__(self, index):
        r"""
        Args:
            index (integer): Index of an image.
        
        Returns:
            img, target (tuple): ``target`` is dict of annotations.
        """
        ins_name = self.ins_names[index]
        ins_info = self.ins_dict[ins_name]

        # parse image and annotations
        img = ops.read_image(ins_info['img_file'])
        target = ins_info['target']
        if hasattr(self, 'transforms') and self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.ins_dict)
    
    @abc.abstractmethod
    def _construct_ins_dict(self, **kwargs):
        raise NotImplementedError
