from typing import Sequence, List
import logging

import tensorflow as tf
import numpy as np

from tf_pipeline.utils.registry.post_proc import POST_PROC

LOG = logging.getLogger()
TAG_NAME = "[PostProc]"

def nms(heat, kernel=3):
    hmax = tf.keras.layers.MaxPooling2D(kernel, 1, padding="same")(heat)
    keep = tf.cast(tf.equal(heat, hmax), tf.float32)
    return heat * keep


def topk(hm, k=100):
    batch, height, width, cat = tf.shape(hm)[0], tf.shape(
        hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    scores = tf.reshape(hm, (batch, -1))
    topk_scores, topk_inds = tf.nn.top_k(scores, k=k)

    topk_clses = topk_inds % cat
    topk_xs = tf.cast(topk_inds // cat % width, tf.float32)
    topk_ys = tf.cast(topk_inds // cat // width, tf.float32)
    topk_inds = tf.cast(
        topk_ys * tf.cast(width, tf.float32) + topk_xs, tf.int32)
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def filter_detections(detections, score_threshold=0.3, nms_iou_threshold=0.5) -> tf.Tensor:
    """
    Remove detections with a low probability, run Non-maximum Suppression. If set, apply max size to all boxes.

    :param detections: Tensor with detections in form [xmin, ymin, xmax, ymax, label, probability].
    :param score_threshold: Minimal probability of a detection to be included in the result.
    :param nms_iou_threshold: Threshold for deciding whether boxes overlap too much with respect to IOU.
    :param max_size: Max size to strip the given bounding boxes, default: None = no stripping.
    :param class_nms: If True use nms per classes (default False)
    :return: Filtered bounding boxes.
    """
    mask = detections[:, 4] >= score_threshold
    result = tf.boolean_mask(detections, mask)
    labels, scores = result[:, 5], result[:, 4]
    xyxy_bboxes = result[:, 0:4]

    # Convert to YXYX (as required by tf.image.non_max_suppression)
    bboxes = xyxy_bboxes.numpy()
    bboxes[:, 0] = xyxy_bboxes[:, 1]
    bboxes[:, 2] = xyxy_bboxes[:, 3]
    bboxes[:, 1] = xyxy_bboxes[:, 0]
    bboxes[:, 3] = xyxy_bboxes[:, 2]

    if tf.shape(bboxes)[0] == 0:
        return tf.constant([], shape=(0, 6))

    max_objects = tf.shape(result)[0]

    selected_indices = tf.image.non_max_suppression(
        bboxes, scores, max_objects, iou_threshold=nms_iou_threshold)
    selected_boxes = tf.gather(bboxes, selected_indices)
    selected_labels = tf.gather(labels, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)

    # Convert to XYXY
    selected_boxes_xyxy = selected_boxes.numpy()
    selected_boxes_xyxy[:, 0] = selected_boxes[:, 1]
    selected_boxes_xyxy[:, 2] = selected_boxes[:, 3]
    selected_boxes_xyxy[:, 1] = selected_boxes[:, 0]
    selected_boxes_xyxy[:, 3] = selected_boxes[:, 2]

    detections = tf.concat(
        [selected_boxes_xyxy, tf.expand_dims(selected_scores, axis=1), tf.expand_dims(selected_labels, axis=1)], axis=1
    )
    return detections


@POST_PROC.register_module()
class CenterNetPostProc(object):
    def __init__(self, score_threshold: float, iou_threshold: float, max_objects: int = 100, down_ratio: int = 4):

        self.pp_name = "[CenterNetPostProc]"

        LOG.info(f"{TAG_NAME} {self.pp_name} score_threshold: {score_threshold}")
        LOG.info(f"{TAG_NAME} {self.pp_name} iou_threshold: {iou_threshold}")
        LOG.info(f"{TAG_NAME} {self.pp_name} max_objects: {max_objects}")
        LOG.info(f"{TAG_NAME} {self.pp_name} down_ratio: {down_ratio}")

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_objects = max_objects
        self.down_ratio = down_ratio

    def decode(self, heatmap: tf.Tensor, wh: tf.Tensor, reg: tf.Tensor, flip_xy: bool = False):
        batch, height, width, channels = tf.shape(heatmap)[0], tf.shape(heatmap)[1], \
            tf.shape(heatmap)[2], tf.shape(heatmap)[3]
        heatmap = nms(heatmap)
        scores, inds, classes, ys, xs = topk(heatmap, self.max_objects)

        ys = tf.expand_dims(ys, axis=-1)
        xs = tf.expand_dims(xs, axis=-1)

        reg = tf.reshape(reg, (batch, -1, tf.shape(reg)[-1]))
        reg = tf.gather(reg, inds, axis=1, batch_dims=-1)
        if flip_xy:
            xs, ys = xs + reg[..., 1:2], ys + reg[..., 0:1]
        else:
            xs, ys = xs + reg[..., 0:1], ys + reg[..., 1:2]

        wh = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
        wh = tf.gather(wh, inds, axis=1, batch_dims=-1)

        classes = tf.cast(tf.expand_dims(classes, axis=-1), tf.float32)
        scores = tf.expand_dims(scores, axis=-1)

        wh = tf.math.abs(wh)

        if flip_xy:
            xmin = xs - wh[..., 1:2] / 2
            xmax = xs + wh[..., 1:2] / 2
            ymin = ys - wh[..., 0:1] / 2
            ymax = ys + wh[..., 0:1] / 2
        else:
            xmin = xs - wh[..., 0:1] / 2
            ymin = ys - wh[..., 1:2] / 2
            xmax = xs + wh[..., 0:1] / 2
            ymax = ys + wh[..., 1:2] / 2

        bboxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)
        detections = tf.concat([bboxes, scores, classes], axis=-1)
        return detections

    def __call__(self, model_preds: Sequence[tf.Tensor]) -> List[np.ndarray]:
        detections = self.decode(
            model_preds[0], model_preds[1], model_preds[2], flip_xy=False)
        N, H, W, C = model_preds[0].shape
        filtered_dets = []
        for det in detections.numpy():
            det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, W)
            det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, H)
            filt_dets = filter_detections(det, self.score_threshold, self.iou_threshold).numpy().astype('double')
            filt_dets[:, 0:4] *= self.down_ratio
            filtered_dets.append(filt_dets)
        return filtered_dets
