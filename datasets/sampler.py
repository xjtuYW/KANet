import numpy as np
import torch
from torch.utils.data import Sampler
from copy import deepcopy
import random
import pdb

__all__ = ['inc_sampler', 'Auto_Task_Sampler', 'ClsSampler', 'CategoriesSamplerAlignInc', 'ClassAwareSampler']


# sample n_iter tasks: int iter loop, each task may have different cls in each sample procedure
class CategoriesSampler(Sampler):
    def __init__(self,
                 label,
                 n_tasks,  # num of n_tasks we will go to sample
                 n_way,  # num of cls in each task
                 n_shot,  # num of samples in each cls used to get proto
                 n_query  # num of samples in each cls used to get result
                 ):
        self.n_tasks = n_tasks
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        label = np.array(label)
        self.m_ind = []

        unique = np.sort(np.unique(label))  # delete rep ele, then rank by ascent order
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind) 
        

    def __len__(self):
        return self.n_tasks
        
    # batch->task->n_way,k_shot
    def __iter__(self):
        for i in range(self.n_tasks):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_way]  #
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                batch.append(l[pos[:self.n_shot + self.n_query]])  # include each label's position
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

    
# sampler used for meta-training
class Auto_Task_Sampler(Sampler):
    def __init__(self, 
                 label,
                 n_way,  # num of cls in each task
                 n_shot,  # num of samples in each cls used to get proto
                 n_query  # num of samples in each cls used to get result)
                 ):
        self.data_nums = 0
        self.way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        if n_query != 0:
            self.shots = [self.n_shot, self.n_query]
        else:
            self.shots = [self.n_shot]

        class2id = {}
        for i, class_id in enumerate(label):
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(i)
            self.data_nums += 1

        self.class2id = class2id
    
    def __len__(self):
        return self.data_nums // (self.way*(self.n_shot+self.n_query))

    def __iter__(self):

        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])

        while len(temp_class2id) >= self.way:

            id_list = []

            list_class_id = list(temp_class2id.keys())

            pcount = np.array([len(temp_class2id[class_id]) for class_id in list_class_id])

            batch_class_id = np.random.choice(list_class_id, size=self.way, replace=False, p=pcount / sum(pcount))

            for shot in self.shots:
                for class_id in batch_class_id:
                    for _ in range(shot):
                        id_list.append(temp_class2id[class_id].pop())

            for class_id in batch_class_id:
                if len(temp_class2id[class_id]) < sum(self.shots):
                    temp_class2id.pop(class_id)

            yield id_list

      
# e.g. 
class ClsSampler(Sampler):
    # label of class start from 0 to (maxcls-1)
    def __init__(self, label):
        self.inds = {}
        for i, class_id in enumerate(label):
            if class_id not in self.inds:
                self.inds[class_id] = []
            self.inds[class_id].append(i)
    
    def __len__(self):
        return len(self.inds.keys())

    def __iter__(self):
        temp_inds = deepcopy(self.inds)
        for k in temp_inds.keys():
            id_list = temp_inds[k]
            yield id_list


class inc_sampler(Sampler):
    def __init__(self, 
                 label,
                 n_way,  # num of cls in each task
                 n_shot  # num of samples in each cls used to get proto
                 ):
        self.way    = n_way
        self.shot   = n_shot
  
        class2id = {}
        for i, class_id in enumerate(label):
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(i)
         
        self.class2id = {}
        sort_keys = sorted(class2id)
        for k in sort_keys:
            self.class2id[k] = class2id[k]

    def __len__(self):
        return len(self.class2id) // self.way

    def __iter__(self):
        temp_class2id = deepcopy(self.class2id)
        for class_id in temp_class2id:
            np.random.shuffle(temp_class2id[class_id])

        while len(temp_class2id) >= self.way:
            id_list = []

            list_class_id = list(temp_class2id.keys())
            batch_class_id = list_class_id[:self.way]
            for class_id in batch_class_id:
                if len(temp_class2id[class_id]) < self.shot:
                    valid_nums = len(temp_class2id[class_id])
                else:
                    valid_nums = self.shot
                for _ in range(valid_nums):
                    id_list.append(temp_class2id[class_id].pop())
                
            for class_id in batch_class_id:
                temp_class2id.pop(class_id)
            yield id_list



# sample n_iter tasks: int iter loop, each task may have different cls in each sample procedure
# consists of both few-shot and conventional tasks
class CategoriesSamplerAlignInc(Sampler):
    def __init__(self,
                 label,
                 n_tasks,  # num of n_tasks we will go to sample
                 n_way,  # num of cls in each task
                 n_shot,  # num of samples in each cls used to get proto
                 n_query,  # num of samples in each cls used to get result
                 batch_size = 128
                 ):
        self.n_tasks = n_tasks
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.batch_size = batch_size
        label = np.array(label)
        self.m_ind = []
        unique = np.sort(np.unique(label))  # delete rep ele, then rank by ascent order
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind) 
        

    def __len__(self):
        return self.n_tasks
        
    # batch->task->n_way,k_shot
    def __iter__(self):
        for i in range(self.n_tasks):
            batch = []
            classes      = torch.randperm(len(self.m_ind))
            inc_classes  = classes[:self.n_way]  #
            base_classes = classes[self.n_way:]  #
            for c in inc_classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                batch.append(l[pos[:self.n_shot + self.n_query]])  # include each label's position
            batch = torch.stack(batch).t().reshape(-1)

            # sample base data
            base_data = []
            for c in base_classes:
                tmp_data = self.m_ind[c.item()]
                base_data.append(tmp_data)
            # base_data  = torch.stack(base_data).reshape(-1)
            base_data  = torch.cat(base_data, dim=-1).reshape(-1)
            batch_base = base_data[torch.randperm(base_data.shape[0])[:self.batch_size]]
            batch = torch.cat((batch, batch_base), dim=0)
            
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch


"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


##################################
## Class-aware sampling, partly implemented by frombeijingwithlove
##################################

class RandomCycleIter:
    
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]
    
def class_aware_sample_generator (cls_iter, data_iter_list, n, num_samples_cls=1):

    i = 0
    j = 0
    while i < n:
        
#         yield next(data_iter_list[next(cls_iter)])
        
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

class ClassAwareSampler (Sampler):
    
    def __init__(self, data_source, num_samples_cls=1):
        num_classes         = len(np.unique(data_source.examplar_label))
        self.class_iter     = RandomCycleIter(range(num_classes))
        cls_data_list       = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.examplar_label):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples    = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)
    
    def __len__ (self):
        return self.num_samples

