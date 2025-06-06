import tensorflow as tf
import numpy as np
import cv2


class MeanIoU(object):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def mean_iou(self, y_true, y_pred):
        return tf.numpy_function(self.mean_iou_numpy, [y_true, y_pred], tf.float32)

    def mean_iou_numpy(self, y_true, y_pred):
        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()

        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2
        )
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape(
            (self.num_classes, self.num_classes)
        )

        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 1

        return np.mean(iou).astype(np.float32)

class MeanBJ(object):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def mean_bj(self, y_true, y_pred):
        return tf.numpy_function(self.mean_bj_numpy, [y_true, y_pred], tf.float32)

    def mean_bj_numpy(self, y_true, y_pred):
        target = np.argmax(y_true, axis=-1)
        predicted = np.argmax(y_pred, axis=-1)

        bj_scores = []

        for cls in range(self.num_classes):
            # Extract object boundaries
            true_boundary = cv2.Canny((target == cls).astype(np.uint8) * 255, 100, 200)
            pred_boundary = cv2.Canny((predicted == cls).astype(np.uint8) * 255, 100, 200)

            # Ensure same shape for intersection calculation
            pred_boundary = cv2.resize(pred_boundary, (true_boundary.shape[1], true_boundary.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

            # Compute intersection and union
            intersection = np.logical_and(pred_boundary, true_boundary).sum()
            union = np.logical_or(pred_boundary, true_boundary).sum()

            if union > 0:
                bj_scores.append(intersection / union)

        return np.mean(bj_scores).astype(np.float32) if bj_scores else np.float32(0.0)  # Avoid empty arrays