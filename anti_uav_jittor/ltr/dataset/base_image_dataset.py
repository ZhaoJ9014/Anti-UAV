
from ltr.data.image_loader import jpeg4py_loader
import jittor as jt
from jittor.dataset import Dataset


class BaseImageDataset(Dataset):
    """ 基于Jittor框架的图像数据集基类 """

    def __init__(self, name, root, image_loader=jpeg4py_loader):
        """
        args:
            root - 数据集的根路径
            image_loader (jpeg4py_loader) - 用于读取图像的函数。默认使用jpeg4py (https://github.com/ajkxyz/jpeg4py)
        """
        self.name = name
        self.root = root
        self.image_loader = image_loader

        self.image_list = []     # 包含图像序列的列表。
        self.class_list = []

    def __len__(self):
        """ 返回数据集的大小
        returns:
            int - 数据集中的样本数量
        """
        return self.get_num_images()

    def __getitem__(self, index):
        """ 实现Jittor Dataset接口，用于数据加载
        """
        # 这里需要根据实际的数据集结构和需求实现数据的读取和预处理
        raise NotImplementedError

    def get_name(self):
        """ 数据集的名称

        returns:
            string - 数据集的名称
        """
        raise NotImplementedError

    def get_num_images(self):
        """ Number of sequences in a dataset

        returns:
            int - number of sequences in the dataset."""
        return len(self.image_list)

    def has_class_info(self):
        return False

    def get_class_name(self, image_id):
        return None

    def get_num_classes(self):
        return len(self.class_list)

    def get_class_list(self):
        return self.class_list

    def get_images_in_class(self, class_name):
        raise NotImplementedError

    def has_segmentation_info(self):
        return False

    def get_image_info(self, seq_id):
        """ Returns information about a particular image,

        args:
            seq_id - index of the image

        returns:
            Dict
            """
        raise NotImplementedError

    def get_image(self, image_id, anno=None):
        """ Get a image

        args:
            image_id      - index of image
            anno(None)  - The annotation for the sequence (see get_sequence_info). If None, they will be loaded.

        returns:
            image -
            anno -
            dict - A dict containing meta information about the sequence, e.g. class of the target object.

        """
        raise NotImplementedError