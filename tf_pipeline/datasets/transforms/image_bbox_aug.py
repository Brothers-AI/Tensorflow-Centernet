import logging
from typing import Tuple, List

import cv2
import numpy as np

from tf_pipeline.utils.registry.transforms import TRANSFORMS
from tf_pipeline.datasets.transforms.default import Transforms, TAG_NAME

LOG = logging.getLogger()


@TRANSFORMS.register_module()
class RandomHFlip(Transforms):
    def __init__(self, prob: float = 0.3):
        super(RandomHFlip, self).__init__()

        self.prob = prob
        LOG.info(f"{TAG_NAME} [RandomHFlip]: prob -> {prob}")

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            flipped_image = cv2.flip(image, 1)
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = image.shape[1] - bboxes[:, [2, 0]]
            flipped_image = np.ascontiguousarray(flipped_image)
            bboxes = bboxes.astype(np.float32).reshape(-1, 4)
            return flipped_image, bboxes, labels
        else:
            return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomCropPadded(Transforms):
    def __init__(self, prob: float = 0.5, crop_size: List[int] = [384, 576]):
        super(RandomCropPadded, self).__init__()
        self.prob = prob
        self.crop_size = crop_size
        LOG.info(
            f"{TAG_NAME} [RandomCropPadded]: prob -> {prob}, crop_size -> {crop_size}")

    def get_params(self, image: np.ndarray):
        H, W, C = image.shape
        ch, cw = self.crop_size

        if H < ch or W < cw:
            raise ValueError(
                f"Required crop size {(ch, cw)} is larger than input image size {(H, W)}")

        if H == ch and W == cw:
            return 0, 0, H, W

        i = np.random.randint(0, H - ch + 1)
        j = np.random.randint(0, W - cw + 1)
        return i, j, ch, cw

    def filter_bboxes_labels(self, bboxes: np.ndarray, labels: np.ndarray, i: int, j: int, ch: int, cw: int):

        def is_point_inside_rectangle(x1, y1, x2, y2, x, y):
            if (x > x1 and x < x2 and y > y1 and y < y2):
                return True
            else:
                return False

        def clamp_bbox(bbox, start_x: int, start_y: int, end_x: int, end_y: int):
            x1, y1, x2, y2 = bbox
            if (is_point_inside_rectangle(start_x, start_y, end_x, end_y, x1, y1) or
                    is_point_inside_rectangle(start_x, start_y, end_x, end_y, x2, y2)):
                x1 = max(x1, start_x)
                y1 = max(y1, start_y)
                x2 = min(x2, end_x)
                y2 = min(y2, end_y)
                flag = True
            else:
                x1, y1, x2, y2 = -1, -1, -1, -1
                flag = False
            return flag, x1, y1, x2, y2

        updated_bboxes = []
        updated_labels = []

        for idx, bbox in enumerate(bboxes):
            flag, x1, y1, x2, y2 = clamp_bbox(bbox, j, i, j + cw, i + ch)
            if not flag:
                continue
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            updated_bboxes.append(bbox)
            updated_labels.append(labels[idx])

        updated_bboxes = np.array(updated_bboxes, dtype=np.float32)
        updated_labels = np.array(updated_labels, dtype=labels.dtype)

        return updated_bboxes, updated_labels

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            img_h, img_w, img_c = image.shape
            i, j, ch, cw = self.get_params(image)

            # Crop the image
            image = image[i: i + ch, j: j + cw, :]

            # Get the filtered bboxes and Labels
            corrected_bboxes, corrected_labels = self.filter_bboxes_labels(
                bboxes, labels, i, j, ch, cw)

            # pad the image
            padded_img = np.zeros((img_h, img_w, img_c)).astype(image.dtype)
            padded_img[i: i + ch, j: j + cw, :] = image
            padded_img = np.ascontiguousarray(padded_img)
            corrected_bboxes = corrected_bboxes.astype(np.float32).reshape(-1, 4)
            corrected_labels = corrected_labels.astype(np.int32).reshape(-1, 1)
            return padded_img, corrected_bboxes, corrected_labels
        else:
            return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomCropResized(Transforms):
    def __init__(self, prob: float = 0.5, crop_size: List[int] = [384, 576], interpolation=cv2.INTER_LINEAR):
        super(RandomCropResized, self).__init__()

        self.prob = prob
        self.crop_size = crop_size
        self.interpolation = interpolation
        LOG.info(
            f"{TAG_NAME} [RandomCropResized]: prob -> {prob}, crop_size -> {crop_size}, interpolation -> {interpolation}")

    def get_params(self, image: np.ndarray):
        H, W, C = image.shape
        ch, cw = self.crop_size

        if H < ch or W < cw:
            raise ValueError(
                f"Required crop size {(ch, cw)} is larger than input image size {(H, W)}")

        if H == ch and W == cw:
            return 0, 0, H, W

        i = np.random.randint(0, H - ch + 1)
        j = np.random.randint(0, W - cw + 1)
        return i, j, ch, cw

    def filter_bboxes_labels(self, bboxes: np.ndarray, labels: np.ndarray, i: int, j: int, ch: int, cw: int):

        def is_point_inside_rectangle(x1, y1, x2, y2, x, y):
            if (x > x1 and x < x2 and y > y1 and y < y2):
                return True
            else:
                return False

        def clamp_bbox(bbox, start_x: int, start_y: int, end_x: int, end_y: int):
            x1, y1, x2, y2 = bbox
            if (is_point_inside_rectangle(start_x, start_y, end_x, end_y, x1, y1) or
                    is_point_inside_rectangle(start_x, start_y, end_x, end_y, x2, y2)):
                x1 = max(x1, start_x)
                y1 = max(y1, start_y)
                x2 = min(x2, end_x)
                y2 = min(y2, end_y)
                flag = True
            else:
                x1, y1, x2, y2 = -1, -1, -1, -1
                flag = False
            return flag, x1, y1, x2, y2

        updated_bboxes = []
        updated_labels = []
        start_array = np.array([j, i, j, i], dtype=np.float32)

        for idx, bbox in enumerate(bboxes):
            flag, x1, y1, x2, y2 = clamp_bbox(bbox, j, i, j + cw, i + ch)
            if not flag:
                continue
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32) - start_array
            updated_bboxes.append(bbox)
            updated_labels.append(labels[idx])

        updated_bboxes = np.array(updated_bboxes, dtype=np.float32)
        updated_labels = np.array(updated_labels, dtype=labels.dtype)

        return updated_bboxes, updated_labels

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            img_h, img_w, img_c = image.shape
            i, j, ch, cw = self.get_params(image)

            # Crop the image
            cropped_image = image[i: i + ch, j: j + cw, :]

            # Get the filtered bboxes and Labels
            corrected_bboxes, corrected_labels = self.filter_bboxes_labels(
                bboxes, labels, i, j, ch, cw)

            # Resize the image
            image = cv2.resize(cropped_image, (img_w, img_h),
                               interpolation=self.interpolation)

            # Scales
            scale_x = img_w / cw
            scale_y = img_h / ch

            # Correct the Bboxes by scaling
            if len(corrected_bboxes) > 0:
                corrected_bboxes = corrected_bboxes * \
                    np.array([scale_x, scale_y, scale_x,
                             scale_y], dtype=np.float32)
            image = np.ascontiguousarray(image)
            corrected_bboxes = corrected_bboxes.astype(np.float32).reshape(-1, 4)
            corrected_labels = corrected_labels.astype(np.int32).reshape(-1, 1)
            return image, corrected_bboxes, corrected_labels
        else:
            return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomRotate(Transforms):
    def __init__(self, prob: 0.5, area_threshold: float = 200):
        super(RandomRotate, self).__init__()

        self.prob = prob
        LOG.info(
            f"{TAG_NAME} [RandomRotate]: prob -> {prob}, area_threshold -> {area_threshold}")

        self.area_threshold = area_threshold
        angles = list(range(-40, -9, 5))
        angles.extend(list(range(10, 41, 5)))
        self.angles_list = angles

    def rotate_image(self, image: np.ndarray, angle: float):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX - 0.5, cY - 0.5), angle, 1.0)
        # Rotate Image
        rot_image = cv2.warpAffine(image, M, (w, h))
        return rot_image, M, h, w

    def rotate_bboxes_and_fit(self, bboxes: np.ndarray, labels: np.ndarray, M, width: int, height: int):
        # Reshape to N x 4
        bboxes = bboxes.reshape(-1, 4)
        xmin, ymin, xmax, ymax = bboxes[:, 0].reshape(-1, 1), bboxes[:, 1].reshape(-1, 1), \
            bboxes[:, 2].reshape(-1, 1), bboxes[:, 3].reshape(-1, 1)
        bbox_corners = np.hstack(
            (xmin, ymin, xmax, ymin, xmin, ymax, xmax, ymax)).reshape(-1, 2)
        corners = np.hstack(
            (bbox_corners, np.ones((bbox_corners.shape[0], 1))))

        rotated_bboxes = np.dot(M, corners.T).T
        rotated_bboxes = rotated_bboxes.reshape(-1, 8)
        x_values = rotated_bboxes[:, [0, 2, 4, 6]]
        y_values = rotated_bboxes[:, [1, 3, 5, 7]]
        xmin = np.min(x_values, 1).reshape(-1, 1)
        ymin = np.min(y_values, 1).reshape(-1, 1)
        xmax = np.max(x_values, 1).reshape(-1, 1)
        ymax = np.max(y_values, 1).reshape(-1, 1)
        rotated_bboxes = np.hstack((xmin, ymin, xmax, ymax))

        assert len(bboxes) == len(labels), f"Boxes and labels are of not same length boxes ({len(bboxes)}), \
                                             labels ({len(labels)})"
        assert len(rotated_bboxes) == len(bboxes), f"Boxes after rotation is not same rotated ({len(rotated_bboxes)}), \
                                                    original ({len(bboxes)})"

        rot_filt_bboxes = []
        filt_labels = []
        for bbox, label in zip(rotated_bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            if ((xmin < 0) and (xmax < 0) or (xmin > width) and (xmax > width) or
                (ymin < 0) and (ymax < 0) or (ymin > height) and (ymax > height)):
                continue

            # Clamp the values
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, width)
            ymax = min(ymax, height)

            # If the bbox is small ignore
            area = (xmax - xmin) * (ymax - ymin)
            if area < self.area_threshold:
                continue

            rot_filt_bboxes.append([xmin, ymin, xmax, ymax])
            filt_labels.append(label)

        rot_filt_bboxes = np.array(rot_filt_bboxes, dtype=np.float32).reshape(-1, 4)
        filt_labels = np.array(filt_labels, dtype=np.int32).reshape(-1, 1).astype(labels.dtype)
        return rot_filt_bboxes, filt_labels

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            angle = np.random.choice(self.angles_list)
            rotated_img, M, h, w = self.rotate_image(image, angle)
            rotated_bboxes, filt_labels = self.rotate_bboxes_and_fit(bboxes, labels, M, w, h)
            rotated_img = np.ascontiguousarray(rotated_img)
            rotated_bboxes = rotated_bboxes.astype(np.float32).reshape(-1, 4)
            filt_labels = filt_labels.astype(np.int32).reshape(-1, 1)
            return rotated_img, rotated_bboxes, filt_labels
        else:
            return image, bboxes, labels
