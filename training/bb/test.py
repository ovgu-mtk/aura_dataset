import os
import argparse
import tensorflow as tf
import numpy as np
import keras_cv
from training.bb.dataloader import BoundingBoxDatasetLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import json
import tempfile

from training.bb.model import create_yolo_model


def parse_args():
    parser = argparse.ArgumentParser(description="Test object detection model with COCO metrics.")
    parser.add_argument('--gpu', type=int, choices=range(0, 4), default=0, help='GPU index to use (0-3). Default: 0')
    parser.add_argument('--img_height', type=int, default=1086, help='Image height. Default: 1086')
    parser.add_argument('--img_width', type=int, default=2046, help='Image width. Default: 2046')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size. Default: 1')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on. Default: test')
    parser.add_argument('--visualization_samples', type=int, default=5,
                        help='Number of samples to visualize. Default: 5')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Confidence score threshold for detections. Default: 0.1')
    parser.add_argument('--merge_classes', type=str, choices=['true', 'false'], default='true',
                        help='Activate/Deactivate merge classes. Default: true')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Number of samples for validation, if None complete dataset is used. Default: None')
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
    """Visualize model predictions on a few samples from the dataset with GT and predictions side by side."""
    for i, batch in enumerate(dataset.take(num_samples)):
        if i >= num_samples:
            break

        images, y_true = batch

        # Make predictions
        # if use infer function no "confidence" value in model output -> no filtering via score_threshold
        # y_pred = infer(model,images)
        y_pred = model.predict(images, verbose=0)

        # Convert predictions to the expected format for visualization
        # Filter by confidence score
        for j in range(len(images)):

            # Filter predictions by confidence score
            filtered_boxes = []
            filtered_classes = []
            filtered_scores = []

            object_counter = 0
            gt_objects = len(y_true["classes"][j])

            for box_idx in range(len(y_pred["boxes"][j])):
                if y_pred["confidence"][j][box_idx] >= score_threshold:
                    object_counter += 1
                    filtered_boxes.append(y_pred["boxes"][j][box_idx])
                    filtered_classes.append(y_pred["classes"][j][box_idx])
                    score = y_pred["confidence"][j][box_idx]
                    filtered_scores.append(score)

            print(f"Sample {i + 1}, Image {j + 1}: Predicted objects: {object_counter} / GT objects: {gt_objects}")
            if object_counter > 0:
                for k, score in enumerate(filtered_scores):
                    print(f"  Predicted object {k + 1} with confidence: {score:.3f}")

            # Create filtered predictions
            filtered_y_pred = {
                "boxes": np.array([filtered_boxes]),
                "classes": np.array([filtered_classes]),
                "confidence": np.array([filtered_scores])
            }

            # Create separate images for GT and predictions
            current_image = images[j:j + 1]

            # Plot Ground Truth
            print(f"\nDisplaying Ground Truth for sample {i + 1}, image {j + 1}:")
            if gt_objects > 0:
                keras_cv.visualization.plot_bounding_box_gallery(
                    current_image,
                    value_range=(0, 255),
                    rows=1, cols=1,
                    y_true=y_true,
                    y_pred=None,  # Only show ground truth
                    scale=5,
                    font_scale=0.7,
                    bounding_box_format="xyxy",
                    class_mapping=class_mapping
                ).show()
            else:
                print("No ground truth objects to display")
                keras_cv.visualization.plot_bounding_box_gallery(
                    current_image,
                    value_range=(0, 255),
                    rows=1, cols=1,
                    scale=5,
                    font_scale=0.7,
                    bounding_box_format="xyxy",
                    class_mapping=class_mapping
                ).show()

            # Plot Predictions
            print(f"Displaying Predictions for sample {i + 1}, image {j + 1}:")
            if object_counter > 0:
                keras_cv.visualization.plot_bounding_box_gallery(
                    current_image,
                    value_range=(0, 255),
                    rows=1, cols=1,
                    y_true=None,  # Only show predictions
                    y_pred=filtered_y_pred,
                    scale=5,
                    font_scale=0.7,
                    bounding_box_format="xyxy",
                    class_mapping=class_mapping
                ).show()
            else:
                print("No predictions above threshold to display")
                keras_cv.visualization.plot_bounding_box_gallery(
                    current_image,
                    value_range=(0, 255),
                    rows=1, cols=1,
                    scale=5,
                    font_scale=0.7,
                    bounding_box_format="xyxy",
                    class_mapping=class_mapping
                ).show()

            print("-" * 50)  # Separator between images



def evaluate_coco_metrics_(model, dataset, score_threshold=0.5, max_samples=None):
    """Evaluate model using COCO metrics with confidence threshold and progress display."""

    # Generate predictions
    coco_predictions = []
    gt_annotations = []
    image_ids = set()
    image_id_counter = 1
    annotation_id = 1
    total_samples = 0

    # Estimate number of batches (only works if dataset is finite, else fallback to simple tqdm)
    total = max_samples if max_samples else sum(1 for _ in dataset)

    progress = tqdm(total=total, desc="Evaluating", unit="img")

    for batch in dataset:
        if max_samples is not None and total_samples >= max_samples:
            break

        images, y_true = batch
        batch_size = images.shape[0]

        if max_samples is not None and total_samples + batch_size > max_samples:
            batch_size = max_samples - total_samples
            images = images[:batch_size]
            for key in y_true:
                y_true[key] = y_true[key][:batch_size]

        y_pred = model.predict(images, verbose=0)

        for i in range(batch_size):
            img_id = image_id_counter
            image_ids.add(img_id)
            image_id_counter += 1

            # GT annotations
            for j in range(len(y_true["classes"][i])):
                box = y_true["boxes"][i][j].numpy()
                x, y, x2, y2 = box
                width, height = x2 - x, y2 - y
                gt_annotations.append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": int(y_true["classes"][i][j]),
                    "bbox": [float(x), float(y), float(width), float(height)],
                    "iscrowd": 0,
                    "area": float(width * height)
                })
                annotation_id += 1

            # Predictions
            for j in range(len(y_pred["boxes"][i])):
                score = y_pred["confidence"][i][j]
                if score < score_threshold:
                    continue
                box = y_pred["boxes"][i][j]
                x, y, x2, y2 = box
                width, height = x2 - x, y2 - y
                coco_predictions.append({
                    "image_id": int(img_id),
                    "category_id": int(y_pred["classes"][i][j]),
                    "bbox": [float(x), float(y), float(width), float(height)],
                    "score": float(score)
                })

        total_samples += batch_size
        progress.update(batch_size)

    progress.close()
    print(f"\nProcessed {total_samples} samples for COCO evaluation")

    # Create fake COCO format
    images_coco = [{"id": int(img_id), "width": 2046, "height": 1086} for img_id in image_ids]
    categories_coco = [{"id": i, "name": str(i)} for i in range(1, 91)]  # Adjust if needed

    gt_coco = {
        "images": images_coco,
        "annotations": gt_annotations,
        "categories": categories_coco
    }

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as gt_file:
        json.dump(gt_coco, gt_file)
        gt_file_path = gt_file.name

    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as pred_file:
        json.dump(coco_predictions, pred_file)
        pred_file_path = pred_file.name

    # Run COCO evaluation
    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(pred_file_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {metric: float(coco_eval.stats[i]) for i, metric in enumerate([
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"
    ])}

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

    # Convert string values to boolean
    args.merge_classes = args.merge_classes.lower() == 'true'


    # Load test dataset
    print(f"Loading {args.split} dataset...")
    test_loader = BoundingBoxDatasetLoader(image_size=(args.img_height, args.img_width),
                                            batch_size=args.batch_size,
                                            split=args.split,
                                            merge_classes=args.merge_classes,
                                            augmentation=False  # No augmentation for test/val
                                           )
    test_ds = test_loader.get_datasets()


    # Print all arguments
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Get model
    model_name = "yolo_v8_xs_12_classes_2"
    #model_name = "yolo_v8_xl_12_classes"


    if model_name == "yolo_v8_xs_12_classes":
        model_backbone = "yolo_v8_xs_backbone_coco"
    elif model_name == "yolo_v8_xl_12_classes":
        model_backbone = "yolo_v8_xl_backbone_coco"
    elif model_name == "yolo_v8_ö_12_classes":
        model_backbone = "yolo_v8_l_backbone_coco"
    else:
        raise ValueError("not a valid model_name")


    model_path = f"model/{model_name}.keras"
    print(f"Loading model from {model_path}...")
    # Create YOLO detector with pre-trained backbone
    model = create_yolo_model(dataloader=test_loader, backbone=model_backbone)

    # load weights
    model.load_weights(model_path)

    # Get class mapping for visualization
    class_mapping = test_loader.class_mapping_merged

    # Visualize some predictions
    #print(f"Visualizing {args.visualization_samples} samples...")
    #visualize_predictions(model, test_ds, class_mapping,
    #                      num_samples=args.visualization_samples,
    #                      score_threshold=args.score_threshold)

    # Evaluate COCO metrics
    print("\nEvaluating COCO metrics (this may take a while)...")
    results = evaluate_coco_metrics_(model, test_ds, score_threshold=0.5, max_samples=None)

    # Print results
    report = format_coco_results(results)
    print(report)

    # Save results to file
    with open(f"{os.path.splitext(model_path)[0]}_coco_metrics.txt", "w") as f:
        f.write(report)

    print(f"Results saved to {os.path.splitext(model_path)[0]}_coco_metrics.txt")


