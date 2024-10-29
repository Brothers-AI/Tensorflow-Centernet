import logging
from typing import Tuple

import cv2
import numpy as np

from tf_pipeline.utils.registry.transforms import TRANSFORMS
from tf_pipeline.datasets.transforms.default import Transforms, TAG_NAME

LOG = logging.getLogger()


@TRANSFORMS.register_module()
class RandomColorSpace(Transforms):
    def __init__(self, prob: float = 0.5):
        super(RandomColorSpace, self).__init__()

        LOG.info(f"{TAG_NAME} [RandomColorSpace]: prob -> {prob}")
        self.prob = prob

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            image = image[:, :, ::-1]
            image = np.ascontiguousarray(image)
        return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomBlur(Transforms):
    def __init__(self, prob: float = 0.5):
        super(RandomBlur, self).__init__()

        LOG.info(f"{TAG_NAME} [RandomBlur]: prob -> {prob}")
        self.prob = prob

        self.funs = [self.avg_blur, self.gaussian_blur]

    def avg_blur(self, image):
        return cv2.blur(image, (5, 5))

    def gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (7, 7), 0)

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            idx = np.random.randint(0, len(self.funs))
            image = self.funs[idx](image)
            image = np.ascontiguousarray(image)
        return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomSharping(Transforms):
    def __init__(self, prob: float = 0.5):
        super(RandomSharping, self).__init__()

        LOG.info(f"{TAG_NAME} [RandomSharping]: prob -> {prob}")
        self.prob = prob

        # Sharping kernel
        self.kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            image = cv2.filter2D(image, -1, self.kernel)
            image = np.ascontiguousarray(image)
        return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomHueSaturation(Transforms):
    def __init__(self, prob: float = 0.5):
        super(RandomHueSaturation, self).__init__()

        LOG.info(f"{TAG_NAME} [RandomHueSaturation]: prob -> {prob}")
        self.prob = prob

        self.funs = [self.hue, self.saturation]

    def hue(self, image: np.ndarray) -> np.ndarray:
        rand_hue = np.random.randint(10, 30)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 0] += rand_hue
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return image

    def saturation(self, image: np.ndarray) -> np.ndarray:
        rand_sat = np.random.randint(10, 30)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 1] += rand_sat
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return image

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            idx = np.random.randint(0, len(self.funs))
            image = self.funs[idx](image)
            image = np.ascontiguousarray(image)
        return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomBrightnessContrast(Transforms):
    def __init__(self, prob: float = 0.5):
        super(RandomBrightnessContrast, self).__init__()

        LOG.info(f"{TAG_NAME} [RandomBrightnessContrast]: prob -> {prob}")
        self.prob = prob

    def brightness_image(self, image: np.ndarray) -> np.ndarray:
        brightness = np.random.randint(100, 150)
        brightness = int((brightness - 0) *
                         (255 - (-255)) / (510 - 0) + (-255))

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                max = 255
            else:
                shadow = 0
                max = 255 + brightness
            alpha = (max - shadow) / 255
            gamma = shadow
            bright_image = cv2.addWeighted(image, alpha, image, 0, gamma)
        else:
            bright_image = image
        return bright_image

    def contrast_image(self, image: np.ndarray) -> np.ndarray:
        contrast = np.random.randint(100, 150)
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

        if contrast != 0:
            alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            gamma = 127 * (1 - alpha)
            contr_image = cv2.addWeighted(image, alpha, image, 0, gamma)
        else:
            contr_image = image
        return contr_image

    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            image = self.brightness_image(image)
            image = self.contrast_image(image)
            image = np.ascontiguousarray(image)
        return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomSolarize(Transforms):
    def __init__(self, prob: float = 0.5, threshold: int = 127):
        super(RandomSolarize, self).__init__()

        LOG.info(f"{TAG_NAME} [RandomSolarize]: prob -> {prob}")
        self.prob = prob
        self.threshold = int(threshold)
    
    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            idx = image >= self.threshold
            image[idx] = 255 - image[idx]
            image = np.ascontiguousarray(image)
        return image, bboxes, labels

@TRANSFORMS.register_module()
class RandomGamma(Transforms):
    def __init__(self, prob: float = 0.5, gamma_limit: Tuple[int] = (80, 150)):
        super(RandomGamma, self).__init__()

        LOG.info(f"{TAG_NAME} [RandomGamma]: prob -> {prob}")
        self.prob = prob
        self.gamma_limit = tuple(gamma_limit)
    
    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            assert image.dtype == np.uint8, f"Expects data type to be uint8"
            gamma = np.random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0
            table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
            image = cv2.LUT(image, table.astype(np.uint8))
            image = np.ascontiguousarray(image)
        return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomHSV(Transforms):
    def __init__(self, prob: float = 0.2, h_gain: float = 0.5,
                 s_gain: float = 0.5, v_gain: float = 0.5):
        super(RandomHSV, self).__init__()

        LOG.info(f"{TAG_NAME} [RandomHSV]: prob -> {prob}, h_gain -> {h_gain} \
                s_gain -> {s_gain}, v_gain -> {v_gain}")
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain
        self.prob = prob
    
    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            if self.h_gain or self.s_gain or self.v_gain:
                r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1  # random gains
                hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
                dtype = image.dtype  # uint8

                x = np.arange(0, 256, dtype=r.dtype)
                lut_hue = ((x * r[0]) % 180).astype(dtype)
                lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
                lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

                im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
                cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed
        return image, bboxes, labels


@TRANSFORMS.register_module()
class RandomHistEqualize(Transforms):
    def __init__(self, clahe: bool = True, prob: bool = 0.2):
        super(RandomHistEqualize, self).__init__()

        LOG.info(f"{TAG_NAME} [RandomHistEqualize]: clahe -> {clahe}, prob -> {prob}")
        self.clahe = clahe
        self.prob = prob
    
    def __call__(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bboxes = bboxes.astype(np.float32).reshape(-1, 4)
        labels = labels.astype(np.int32).reshape(-1, 1)
        if np.random.random() < self.prob:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            if self.clahe:
                c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                yuv[:, :, 0] = c.apply(yuv[:, :, 0])
            else:
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
            image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return image, bboxes, labels
    