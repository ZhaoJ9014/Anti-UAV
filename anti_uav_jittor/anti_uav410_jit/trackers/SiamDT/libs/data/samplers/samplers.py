import numpy as np
import copy
import random
from collections import defaultdict
from jittor.dataset import Sampler

from libs.config import registry


__all__ = ['RandomIdentitySampler']


@registry.register_module
class RandomIdentitySampler(Sampler):

    def __init__(self, dataset, num_identities, num_instances):
        self.dataset = dataset
        self.num_identities = num_identities
        self.num_instances = num_instances
        self.index_dict = defaultdict(list)
        for i, (_, val) in enumerate(dataset.ins_dict.items()):
            ins_id = val['target']['ins_id']
            self.index_dict[ins_id].append(i)
        self.ins_ids = list(self.index_dict.keys())

        # estimate length (samples in an epochs)
        self.length = 0
        for ins_id in self.ins_ids:
            n = len(self.index_dict[ins_id])
            if n < self.num_instances:
                n = self.num_instances
            self.length += n - n % self.num_instances
    
    def __iter__(self):
        # construct batch indices for each ins_id
        batch_index_dict = defaultdict(list)
        for ins_id in self.ins_ids:
            indices = copy.deepcopy(self.index_dict[ins_id])
            if len(indices) < self.num_instances:
                indices = np.random.choice(
                    indices, size=self.num_instances, replace=True)
            random.shuffle(indices)

            batch_indices = []
            for index in indices:
                batch_indices.append(index)
                if len(batch_indices) == self.num_instances:
                    batch_index_dict[ins_id].append(batch_indices)
                    batch_indices = []
        
        # construct the final sample indices
        rest_ids = copy.deepcopy(self.ins_ids)
        final_indices = []
        while len(rest_ids) >= self.num_identities:
            selected = random.sample(rest_ids, self.num_identities)
            for ins_id in selected:
                batch_indices = batch_index_dict[ins_id].pop(0)
                final_indices.extend(batch_indices)
                if len(batch_index_dict[ins_id]) == 0:
                    rest_ids.remove(ins_id)
        
        # update length (samples in an epoch)
        self.length = len(final_indices)

        return iter(final_indices)
    
    def __len__(self):
        return self.length
