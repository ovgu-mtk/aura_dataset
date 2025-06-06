import tensorflow as tf
import keras_cv

"""
## COCO Metrics Callback
"""
class EvaluateCOCOMetricsCallback(tf.keras.callbacks.Callback):
    """Callback to evaluate COCO metrics and save best model."""
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )
        self.best_map = -1.0

    def _pad_labels(self, boxes, classes, max_boxes):
        num_boxes = tf.shape(boxes)[0]
        pad_amount = max_boxes - num_boxes

        # Pad boxes to [max_boxes, 4]
        boxes = tf.pad(boxes, [[0, pad_amount], [0, 0]])

        # Pad classes to [max_boxes]
        classes = tf.pad(classes, [[0, pad_amount]])

        return boxes, classes

    def on_epoch_end(self, epoch, logs=None):
        self.metrics.reset_state()
        logs = logs or {}

        max_gt_boxes = 0
        max_pred_boxes = 0

        # First pass: find max number of boxes
        for batch in self.data:
            _, y_true = batch
            y_pred = self.model.predict(batch[0], verbose=0)

            for gt_boxes, pred_boxes in zip(y_true["boxes"], y_pred["boxes"]):
                max_gt_boxes = tf.maximum(max_gt_boxes, tf.shape(gt_boxes)[0])
                max_pred_boxes = tf.maximum(max_pred_boxes, tf.shape(pred_boxes)[0])

        max_boxes = tf.maximum(max_gt_boxes, max_pred_boxes)

        # Second pass: pad and compute metrics
        for batch in self.data:
            images, y_true = batch
            y_pred = self.model.predict(images, verbose=0)

            batch_size = tf.shape(images)[0]
            padded_y_true = {"boxes": [], "classes": []}
            padded_y_pred = {"boxes": [], "classes": [], "confidence": []}

            for i in range(batch_size):
                true_boxes, true_classes = y_true["boxes"][i], y_true["classes"][i]
                pred_boxes, pred_classes = y_pred["boxes"][i], y_pred["classes"][i]

                true_boxes, true_classes = self._pad_labels(true_boxes, true_classes, max_boxes)
                pred_boxes, pred_classes = self._pad_labels(pred_boxes, pred_classes, max_boxes)
                pred_conf = tf.ones_like(pred_classes, dtype=tf.float32)

                padded_y_true["boxes"].append(true_boxes)
                padded_y_true["classes"].append(true_classes)

                padded_y_pred["boxes"].append(pred_boxes)
                padded_y_pred["classes"].append(pred_classes)
                padded_y_pred["confidence"].append(pred_conf)

            padded_y_true = {k: tf.stack(v, axis=0) for k, v in padded_y_true.items()}
            padded_y_pred = {k: tf.stack(v, axis=0) for k, v in padded_y_pred.items()}

            self.metrics.update_state(padded_y_true, padded_y_pred)

        # Final results
        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["MaP"]
        print(f"\nEpoch {epoch + 1}: Mean Average Precision (MaP): {current_map:.4f}")

        if current_map > self.best_map:
            self.best_map = current_map
            print(f"New best mAP: {self.best_map:.4f}")