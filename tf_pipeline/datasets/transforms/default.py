from typing import Tuple, List
import logging

import numpy as np

LOG = logging.getLogger()
TAG_NAME = "[Transforms]"

class Transforms(object):
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError('To be implemented by child class')


class Compose(object):
    def __init__(self, transforms: List[Transforms]):
        self.transforms = transforms

        LOG.info(f"{TAG_NAME} [Compose]: {self.transforms}")

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        for transform in self.transforms:
            image, bboxes, labels = transform(image, bboxes, labels)

        return image, bboxes, labels