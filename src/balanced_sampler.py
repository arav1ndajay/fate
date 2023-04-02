import numpy as np
from torch.utils.data import Sampler
from utils import CustomDataset

def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


class BalancedBatchSampler(Sampler):
    def __init__(self,
                 dataset: CustomDataset, batch_size: int):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.dataset = dataset

        self.anomalies = True
        self.batch_size = batch_size
        self.steps_per_epoch = len(dataset)//self.batch_size

        self.normal_generator = self.random_generator(self.dataset.normal_idx)
        self.outlier_generator = self.random_generator(self.dataset.outlier_idx)
        if self.anomalies:
            self.n_normal = self.batch_size // 2
            self.n_outlier = self.batch_size - self.n_normal
        else:
            self.n_normal = self.batch_size
            self.n_outlier = 0

    @staticmethod
    def random_generator(idx_list):
        while True:
            random_list = np.random.permutation(idx_list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.steps_per_epoch
    
    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_outlier):
                batch.append(next(self.outlier_generator))
            yield batch