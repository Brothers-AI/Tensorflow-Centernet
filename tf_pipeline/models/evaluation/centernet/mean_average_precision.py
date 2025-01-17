"""
Original file comes from Fizyr, but it was heavily modified by us. Structure changed, batch processing added.

Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import tensorflow as tf
from tf_pipeline.models.evaluation.centernet.overlap import compute_overlap


class MAP(object):
    def __init__(self, classes, iou_threshold=0.5, score_threshold=0.05, max_size=None):
        """
        Creates object which calculates mAP (mean Average Precision) for our detection predictions.

        :param classes: how many different object classes do we have
        :param iou_threshold: "Intersection over Union" threshold when we consider the object to be correctly detected
        :param score_threshold: probability from which we consider the detection to be valid
        :param max_size: maximum size used to clip the predicted bounding boxes
        """
        self.classes = classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.false_positives = None
        self.true_positives = None
        self.scores = None
        self.num_annotations = None
        self.max_size = max_size

        self.reset_states()

    def reset_states(self):
        """
        Resets all the accumulated values. Has to be called before every new epoch if we want to reuse this object.
        """
        self.false_positives = [[] for _ in range(self.classes)]
        self.true_positives = [[] for _ in range(self.classes)]
        self.scores = [[] for _ in range(self.classes)]
        self.num_annotations = [0.0 for _ in range(self.classes)]

    def update_state(self, predictions, bboxes, labels):
        """
        Update our results using given predictions for a single image and its ground truth values.

        :param predictions:
        :param bboxes:
        :param labels:
        :return:
        """

        # Reshaping labels to -1
        labels = labels.reshape(-1)

        annotations = self._get_annotations(bboxes, labels, self.classes)

        detections = self._get_detections(
            predictions, self.classes, score_threshold=self.score_threshold)

        self._evaluate_batch([annotations], [detections],
                             self.classes, iou_threshold=self.iou_threshold)

    def result(self):
        average_precisions = {}
        details_per_class = {}
        for label in range(self.classes):
            false_positives_label = np.array(self.false_positives[label])
            true_positives_label = np.array(self.true_positives[label])
            scores_label = np.array(self.scores[label])
            num_annotations_label = self.num_annotations[label]

            # no annotations
            # (we use "non_empty_classes" for the average calculation, no annotation classes are taken into account)
            if num_annotations_label == 0:
                average_precisions[label] = 0, 0
                details_per_class[label] = {}
                continue

            # no annotations or no positive ones, we set average precision to be 0 for this class
            if (len(false_positives_label) + len(true_positives_label)) == 0:
                average_precisions[label] = 0, num_annotations_label
                details_per_class[label] = {"recall": 0}
                continue

            # sort by score
            indices = np.argsort(-scores_label)
            false_positives_label = false_positives_label[indices]
            true_positives_label = true_positives_label[indices]

            # compute false positives and true positives
            false_positives_label = np.cumsum(false_positives_label)
            true_positives_label = np.cumsum(true_positives_label)

            # compute recall and precision
            recall = true_positives_label / num_annotations_label
            precision = true_positives_label / np.maximum(
                true_positives_label +
                false_positives_label, np.finfo(np.float64).eps
            )

            # compute average precision
            average_precision = self._compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations_label

            details_per_class[label] = {
                "recall": recall[-1], "precision": precision[-1]}

        annotations = sum([class_stats[1]
                          for _, class_stats in average_precisions.items()])
        weighted = (
            sum([class_stats[0] * class_stats[1]
                for _, class_stats in average_precisions.items()]) / annotations
            if annotations > 0
            else 0
        )

        non_empty_classes = sum(
            [class_stats[1] > 0 for _, class_stats in average_precisions.items()])
        overall = (
            sum([class_stats[0]
                for _, class_stats in average_precisions.items()]) / non_empty_classes
            if non_empty_classes > 0
            else 0
        )

        return {
            "overall": overall,
            "weighted": weighted,
            "per_class": average_precisions,
            "details": {"per_class": details_per_class},
        }

    def _evaluate_batch(self, annotations_batch, detections_batch, class_num, iou_threshold):
        # process detections and annotations
        for label in range(class_num):
            for detections, annotations in zip(detections_batch, annotations_batch):
                detections, annotations = detections[label], annotations[label]

                self.num_annotations[label] += annotations.shape[0]

                detected_annotations = []
                for d in detections:
                    self.scores[label].append(d[4])

                    if annotations.shape[0] == 0:
                        self.false_positives[label].append(1)
                        self.true_positives[label].append(0)
                        continue

                    overlaps = compute_overlap(
                        np.expand_dims(d, axis=0), annotations)[0]
                    assigned_annotation = np.argmax(overlaps)
                    max_overlap = overlaps[assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        self.false_positives[label].append(0)
                        self.true_positives[label].append(1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        self.false_positives[label].append(1)
                        self.true_positives[label].append(0)

    def _compute_ap(self, recall, precision):
        """Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _unpack_result(self, result, score_threshold):
        mask = result[:, 4] >= score_threshold
        result = tf.boolean_mask(result, mask)
        bboxes = result[:, 0:4]

        if self.max_size is not None:
            bboxes = tf.clip_by_value(
                tf.cast(bboxes, tf.float32), 0.0, float(self.max_size))

        labels = tf.cast(result[:, 5], tf.int32)
        scores = result[:, 4]

        max_objects = result.shape[0]
        selected_indices = tf.image.non_max_suppression(
            bboxes, scores, max_objects, iou_threshold=0.5)
        selected_boxes = tf.gather(bboxes, selected_indices).numpy()
        selected_labels = tf.gather(labels, selected_indices).numpy()
        selected_scores = tf.gather(scores, selected_indices).numpy()

        return selected_boxes, selected_scores, selected_labels

    def _get_detections(self, results, class_num, score_threshold, allresult=True):
        if isinstance(results, tf.Tensor):
            results = results.numpy().astype(np.float64)
        else:
            results = results.astype(np.float64)

        if not allresult:
            return results

        # copy detections to all_detections
        detections = [None for i in range(class_num)]
        for label in range(class_num):
            detections[label] = results[results[:, -1] == label, :-1]

        return detections

    def _get_annotations(self, bboxes, labels, class_num):
        annotations = [None for _ in range(class_num)]
        for label in range(class_num):
            annotations[label] = bboxes[labels == label, :].copy().astype(np.float64)

        return annotations
