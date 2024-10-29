from typing import Dict

import numpy as np
import cv2

def overlap_bbox_onto_image(image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, color_maps: Dict[str, tuple],
                            renormalize: bool = False, mean: np.ndarray = None, std: np.ndarray = None) -> np.ndarray:
    local_image = image.copy()
    local_image = np.ascontiguousarray(local_image)

    if renormalize:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        local_image = (local_image * std) + mean

    if not isinstance(bboxes, np.ndarray):
        bboxes = np.array(bboxes, dtype=np.uint32).reshape(-1, 4)
    else:
        bboxes = bboxes.astype(np.uint32).reshape(-1, 4)

    if not isinstance(labels, np.ndarray):
        labels = np.array(labels, dtype=np.uint32).reshape(-1, 1)
    else:
        labels = labels.astype(np.uint32).reshape(-1, 1)

    for bbox, label in zip(bboxes, labels):
        try:
            local_image = cv2.rectangle(local_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                    color=tuple(color_maps[int(label)].tolist()), thickness=2)
        except ValueError:
            pass
    return local_image