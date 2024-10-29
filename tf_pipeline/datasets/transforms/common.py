from typing import Any, Tuple, Sequence, List, Union
import logging
import math

import cv2
import numpy as np

from tf_pipeline.utils.registry.transforms import TRANSFORMS
from tf_pipeline.datasets.transforms.default import Transforms

LOG = logging.getLogger()
TAG_NAME = "[Transforms]"

@TRANSFORMS.register_module()
class ResizeTo(Transforms):
    def __init__(self, out_width: int, out_height: int):
        super(ResizeTo, self).__init__()
        self.resize_res = (out_width, out_height)

        LOG.info(f"{TAG_NAME} [ResizeTo]: resolution -> {self.resize_res}")

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        in_h, in_w, _ = image.shape
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.reshape(-1, 1)
        if (in_h == self.resize_res[1] and in_w == self.resize_res[0]):
            return image, bboxes, labels
        else:
            resize_image = cv2.resize(
                image, self.resize_res, interpolation=cv2.INTER_LINEAR)
            scale_w = float(self.resize_res[0]) / float(in_w)
            scale_h = float(self.resize_res[1]) / float(in_h)

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * float(scale_w)
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * float(scale_h)
            # TODO: Remove the bboxes and labels which are very small in resized image
            return resize_image, bboxes, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    '''Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio.'''
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

@TRANSFORMS.register_module()
class ResizeAugV1(Transforms):
    def __init__(self, out_width: int, out_height: int, prob: float = 0.5,
                 degrees: float = 0.0, translate: float = 0.1,
                 scale: float = 0.1, shear: float = 10):
        super(ResizeAugV1, self).__init__()

        self.resize_res = (out_width, out_height)
        self.prob = prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

        LOG.info(f"{TAG_NAME} [ResizeAugV1]: resolution -> {self.resize_res}, prob -> {prob} \
                 degrees -> {degrees}, translate -> {translate}, scale -> {scale}, shear -> {shear}")
    
    def get_transform_matrix(self, img_shape):
        new_width, new_height = self.resize_res
        # Center
        C = np.eye(3)
        C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

        # Rotation and Scale
        R = np.eye(3)
        a = np.random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = np.random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * new_width  # x translation (pixels)
        T[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * new_height  # y transla ion (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
        return M, s
    
    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        in_h, in_w, _ = image.shape
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)

        n_b = bboxes.shape[0]
        n_l = labels.shape[0]

        assert n_b == n_l, f"Expected same length for bboxes ({n_b}) and labels ({n_l})"
        if np.random.random() < self.prob:
            M, s = self.get_transform_matrix([in_h, in_w])

            if (M != np.eye(3)).any():  # image changed
                image = cv2.warpAffine(image, M[:2], dsize=self.resize_res, borderValue=(114, 114, 114))

                if len(bboxes) > 0:
                    new = np.zeros((n_b, 4))

                    xy = np.ones((n_b * 4, 3))
                    xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n_b * 4, 2)
                    xy = xy @ M.T
                    xy = xy[:, :2].reshape(n_b, 8)

                    # create new boxes
                    x = xy[:, [0, 2, 4, 6]]
                    y = xy[:, [1, 3, 5, 7]]
                    new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n_b).T

                    # clip
                    new[:, [0, 2]] = new[:, [0, 2]].clip(0, self.resize_res[0])
                    new[:, [1, 3]] = new[:, [1, 3]].clip(0, self.resize_res[1])

                    # filter candidates
                    i = box_candidates(box1=bboxes.T * s, box2=new.T, area_thr=0.1)
                    bboxes = bboxes[i]
                    labels = labels[i]
                    bboxes[:, 0:4] = new[i]

                    bboxes = bboxes.astype(np.float32).reshape(-1, 4)
                    labels = labels.astype(np.int32).reshape(-1, 1)
            return image, bboxes, labels
        else:
            if (in_h == self.resize_res[1] and in_w == self.resize_res[0]):
                return image, bboxes, labels
            else:
                resize_image = cv2.resize(
                    image, self.resize_res, interpolation=cv2.INTER_LINEAR)
                scale_w = float(self.resize_res[0]) / float(in_w)
                scale_h = float(self.resize_res[1]) / float(in_h)

                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * float(scale_w)
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * float(scale_h)
                # TODO: Remove the bboxes and labels which are very small in resized image
                return resize_image, bboxes, labels


@TRANSFORMS.register_module()
class ResizeAugV2(Transforms):
    def __init__(self, out_width: int, out_height: int, prob: float = 0.5):
        super(ResizeAugV2, self).__init__()

        self.resize_res = (int(out_width), int(out_height))
        self.prob = prob

        LOG.info(f"{TAG_NAME} [ResizeAugV2]: resolution -> {self.resize_res}, prob -> {prob}")

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        in_h, in_w, _ = image.shape
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)

        if np.random.random() < self.prob:
            scale_h = self.resize_res[1] / in_h
            scale_w = self.resize_res[0] / in_w
            scale = min(scale_h, scale_w)

            resize_h = int(in_h * scale)
            resize_w = int(in_w * scale)

            scaled_image = cv2.resize(image, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
            padded_image = np.zeros((self.resize_res[1], self.resize_res[0], 3), dtype=np.uint8)

            if np.random.random() < 0.5:
                padded_image[:resize_h, :resize_w, :] = scaled_image
                pad_x = 0
                pad_y = 0
            else:
                padded_image[-resize_h:, -resize_w:, :] = scaled_image
                pad_x = (self.resize_res[0] - resize_w)
                pad_y = (self.resize_res[1] - resize_h)
            
            bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] * scale) + pad_x
            bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] * scale) + pad_y
            bboxes = bboxes.astype(np.float32).reshape(-1, 4)
            return padded_image, bboxes, labels
        else:
            if (in_h == self.resize_res[1] and in_w == self.resize_res[0]):
                return image, bboxes, labels
            else:
                resize_image = cv2.resize(
                    image, self.resize_res, interpolation=cv2.INTER_LINEAR)
                scale_w = float(self.resize_res[0]) / float(in_w)
                scale_h = float(self.resize_res[1]) / float(in_h)

                bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * float(scale_w)
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * float(scale_h)
                # TODO: Remove the bboxes and labels which are very small in resized image
                return resize_image, bboxes, labels


@TRANSFORMS.register_module()
class Normalize(Transforms):
    def __init__(self, mean: Union[List, np.ndarray], std: Union[List, np.ndarray]):
        super(Normalize, self).__init__()

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        LOG.info(f"{TAG_NAME} [Normalize]: mean -> {self.mean}, std -> {self.std}")

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.reshape(-1, 1)
        norm_image = image.astype(np.float32)
        norm_image = (norm_image - self.mean) / self.std
        return norm_image, bboxes, labels
