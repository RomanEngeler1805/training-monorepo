import math
import random


class DataLoader:
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        """return an iterator over the dataset"""
        self.current_idx = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        """return the next batch of data"""
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        # Get batch of indices
        end_idx = min([self.current_idx + self.batch_size, len(self.dataset)])
        batch_indices = self.indices[self.current_idx : end_idx]

        batch = [self.dataset[idx] for idx in batch_indices]

        self.current_idx = end_idx

        return batch

    def __len__(self):
        """return the number of batches in the dataset"""
        return math.ceil(len(self.dataset) / self.batch_size)

    def __reset__(self):
        """reset the dataset to the beginning"""
        self.current_idx = 0
