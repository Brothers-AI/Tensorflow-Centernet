from typing import Sequence, Dict, Any

import numpy as np
import tensorflow as tf

class DefaultDataset(object):
    def __init__(self, model_resolution: Sequence[int], split: str, transforms: Dict, init_lr: float):
        self.model_resolution = model_resolution
        self.split = split
        self.init_lr = init_lr

        # Set the random seed for same random behaviour for each execution of script
        np.random.seed(42)

        if split not in ["train", "val"]:
            raise NotImplementedError(
                f"Not implemented for split type {split}")
        
        # Check the number of transforms
        if len(transforms) == 0:
            raise ValueError(f"Expected atleast 1 transforms, but found 0")
    
    def __getitem__(self, index):
        raise NotImplementedError("To be implemented by child class")
    
    def __len__(self):
        raise NotImplementedError("To be implemented by child class")
    
    @property
    def num_classes(self):
        raise NotImplementedError("To be implemented by child class")
    
    @property
    def val_json_path(self):
        raise NotImplementedError("To be implemented by child class")
    
    @property
    def output_types(self):
        raise NotImplementedError("To be implemented by child class")
    
    @property
    def output_shapes(self):
        raise NotImplementedError("To be implemented by child class")
    
    def __call__(self):
        if self.split == "train":
            # Shuffle the samples each epoch
            self.samples_indices = np.random.permutation(self.samples_indices)

        for index in self.samples_indices:
            yield self.__getitem__(int(index))
        
    
    def scheduler(self, epoch):
        raise NotImplementedError("To be implemented by child class")


class DefaultDatasetV2(tf.keras.utils.Sequence):
    def __init__(self, batch_size: int, shuffle: bool = False):
        super(DefaultDatasetV2, self).__init__()

        self.batch_size = int(batch_size)

        self.shuffle = shuffle

        # indexes
        self.indexes = np.arange(self.dataset_len)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return self.dataset_len // self.batch_size

    @property
    def output_names(self) -> Sequence[str]:
        raise NotImplementedError('To be implemented by child class')
    
    @property
    def output_shapes(self):
        raise NotImplementedError("To be implemented by child class")

    def get_single_data(self, index) -> Dict[str, Any]:
        raise NotImplementedError('To be implemented by child class')

    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, self.dataset_len)

        batch_data = {name : [] for name in self.output_names}

        for idx in range(low, high):
            single_data = self.get_single_data(idx)
            for name, value in single_data.items():
                batch_data[name].append(value)
        
        np_data = {}
        for name, value in batch_data.items():
            np_data[name] = np.array(value).reshape(self.batch_size, *self.output_shapes[name].as_list())
        return np_data

    def on_epoch_end(self):
        # Update indexes
        self.indexes = np.arange(self.dataset_len)
        if self.shuffle:
            # Shuffle after each epoch on train
            np.random.shuffle(self.indexes)
        return