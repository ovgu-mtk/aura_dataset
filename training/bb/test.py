import os
import argparse
import tensorflow as tf
import numpy as np
import keras_cv
from training.bb.dataloader import BoundingBoxDatasetLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Test object detection model with COCO metrics.")
    parser.add_argument('--gpu', type=int, choices=range(0, 4), default=0, help='GPU index to use (0-3). Default: 0')
    parser.add_argument('--img_height', type=int, default=1086, help='Image height. Default: 1086')
    parser.add_argument('--img_width', type=int, default=2046, help='Image width. Default: 2046')
    parser.add_argument('--model_index', type=int, choices=range(0, 4), default=0,
                        help='Model index to name the saved model. Default: 0')
    parser.add_argument('--model_backbone', type=str, default='yolo_v8_s_backbone_coco',
                        help='Model backbone. Options: yolo_v8_xs_backbone_coco, yolo_v8_s_backbone_coco, '
                             'yolo_v8_m_backbone_coco, yolo_v8_l_backbone_coco, yolo_v8_xl_backbone_coco. Default: 0')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size. Default: 1')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on. Default: test')
    parser.add_argument('--visualization_samples', type=int, default=5,
                        help='Number of samples to visualize. Default: 5')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Confidence score threshold for detections. Default: 0.1')
    return parser.parse_args()


def set_gpu(gpu_index):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        except Exception as e:
            print(f"Failed to use GPU {gpu_index}, falling back to GPU 0. Error: {e}")
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)


# fasten inference speed
@tf.function
def infer(model, X_input):
    return model(X_input, training=False)


def visualize_predictions(model, dataset, class_mapping, num_samples=5, score_threshold=0.5):
    """Visualize model predictions on a few samples from the dataset."""
    for i, batch in enumerate(dataset.take(num_samples)):
        if i >= num_samples:
            break

        images, y_true = batch

        # Make predictions
        y_pred = model.predict(images, verbose=0)

        # Convert predictions to the expected format for visualization
        # Filter by confidence score
        for j in range(len(images)):


            # Filter predictions by confidence score
            filtered_boxes = []
            filtered_classes = []
            filtered_scores = []

            object_counter=0
            for box_idx in range(len(y_pred["boxes"][j])):
                if y_pred["confidence"][j][box_idx] >= score_threshold:
                    object_counter+=1
                    print(f"object_counter: {object_counter}")
                    filtered_boxes.append(y_pred["boxes"][j][box_idx])
                    filtered_classes.append(y_pred["classes"][j][box_idx])
                    score= y_pred["confidence"][j][box_idx]
                    filtered_scores.append(score)

                    print(f"predicted object number: {object_counter} "
                          f"with confidence: {score}")

            # Create filtered predictions
            filtered_y_pred = {
                "boxes": np.array([filtered_boxes]),
                "classes": np.array([filtered_classes]),
                "confidence": np.array([filtered_scores])
            }
            if object_counter > 0:
                keras_cv.visualization.plot_bounding_box_gallery(
                    images[j:j + 1],
                    value_range=(0, 255),
                    rows=1, cols=1,
                    y_true=y_true,
                    y_pred=filtered_y_pred,
                    scale=5,
                    font_scale=0.7,
                    bounding_box_format="xyxy",
                    class_mapping=class_mapping
                ).show()
            else:
                keras_cv.visualization.plot_bounding_box_gallery(
                    images[j:j + 1],
                    value_range=(0, 255),
                    rows=1, cols=1,
                    y_true=y_true,
                    scale=5,
                    font_scale=0.7,
                    bounding_box_format="xyxy",
                    class_mapping=class_mapping
                ).show()
                print("no object predicted")


def evaluate_coco_metrics(model, dataset):
    """Evaluate model using COCO metrics."""
    # Create COCO metrics
    metrics = keras_cv.metrics.BoxCOCOMetrics(
        bounding_box_format="xyxy",
        evaluate_freq=1e9
    )

    # Reset metrics
    metrics.reset_state()

    # Process batches
    for i, batch in enumerate(dataset):
        images, y_true = batch
        y_pred = model.predict(images, verbose=0)

        # Update metrics
        metrics.update_state(y_true, y_pred)

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} batches...")
    # Get results
    results = metrics.result()
    return results


def format_coco_results(results):
    """Format COCO results into a readable report."""
    report = "\n===== COCO Metrics Evaluation Report =====\n"

    for metric_name, value in results.items():
        report += f"{metric_name}: {value:.4f}\n"

    return report


if __name__ == "__main__":
    args = parse_args()

    # Set GPU
    set_gpu(args.gpu)

    # Settings from args
    img_size = (args.img_height, args.img_width)
    batch_size = args.batch_size
    model_name = f"{args.model_index}_model"
    model_path = f"model/{model_name}.keras"

    # Load test dataset
    print(f"Loading {args.split} dataset...")
    test_loader = BoundingBoxDatasetLoader(
        image_size=img_size,
        batch_size=batch_size,
        split=args.split,
        augmentation=False  # No augmentation for test/val
    )

    # Get the dataset
    test_ds = test_loader.get_datasets()

    print(f"Loading model from {model_path}...")

    # Create YOLO detector with pre-trained backbone
    model = keras_cv.models.YOLOV8Detector(num_classes=len(test_loader.class_mapping_merged), bounding_box_format="xyxy",
        backbone=keras_cv.models.YOLOV8Backbone.from_preset(args.model_backbone ), fpn_depth=1)

    # load weights
    model.load_weights(model_path)


    # Get class mapping for visualization
    class_mapping = test_loader.class_mapping_merged

    # Visualize some predictions
    print(f"Visualizing {args.visualization_samples} samples...")
    visualize_predictions(model, test_ds, class_mapping,
                          num_samples=args.visualization_samples,
                          score_threshold=args.score_threshold)

    # Evaluate COCO metrics
    print("\nEvaluating COCO metrics (this may take a while)...")
    results = evaluate_coco_metrics(model, test_ds)

    # Print results
    report = format_coco_results(results)
    print(report)

    # Save results to file
    with open(f"{os.path.splitext(model_path)[0]}_coco_metrics.txt", "w") as f:
        f.write(report)

    print(f"Results saved to {os.path.splitext(model_path)[0]}_coco_metrics.txt")