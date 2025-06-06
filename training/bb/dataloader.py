import os
from pathlib import Path
import xml.etree.ElementTree as ET
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import visualization
import collections

import utils.classes as cls


def find_root(start_path, target_name):
    current_path = Path(start_path).resolve()

    # Traverse up the directory tree
    for parent in current_path.parents:
        if parent.name == target_name:
            return parent

    return None  # Return None if no match is found


# paths
dataset_base = str(find_root(__file__, "aura_dataset")) + "/dataset/bb/"

# Define merge groups: new_id -> [old_ids...]
merge_groups = {
    0: [1], # car
    1: [2], # bus
    2: [3], # tram
    3: [4], # transporter
    4: [5], # truck
    5: [6,7,8,9], # bikes and biker
    6: [10,11,12], # emergency vehicles
    7: [13,14,20],  # traffic lights + railroad crossing signal
    8: list(range(15, 19)),  # person
    9: [19],  # dog
    10: list(range(21, 57)),  # traffic signs
    11: [0] + list(range(57, 64)),  # speed bump + traffic markings
}

# Flatten to a merge_map
merge_map = {old_id: new_id for new_id, old_ids in merge_groups.items() for old_id in old_ids}


class BoundingBoxDatasetLoader:
    def __init__(self,
                 dataset_dir=dataset_base,
                 batch_size=1,
                 image_size=(1086, 2046),
                 split_ratio=0.1,
                 split='train',
                 augmentation=True,
                 augmentation_copies=2,
                 augmentation_scale_factor=(0.8, 1.25),
                 merge_classes=True,
                 class_descriptions=cls.class_descriptions_bb,
                 shuffle=False):

        self.dataset_base = dataset_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.split_ratio = split_ratio
        self.split = split
        self.augmentation = augmentation
        self.augmentation_copies = augmentation_copies
        self.augmentation_scale_factor = augmentation_scale_factor
        self.merge_classes = merge_classes
        self.shuffle = shuffle

        # Set up class descriptions if provided, otherwise use an empty list
        self.class_descriptions = class_descriptions if class_descriptions else []

        if self.split == 'train':
            self.dataset_base = os.path.join(self.dataset_base, "train_validate")
        elif self.split == 'val':
            self.dataset_base = os.path.join(self.dataset_base, "train_validate")
        elif self.split == 'test':
            self.dataset_base = os.path.join(self.dataset_base, "test")
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Set up class mappings
        self.setup_class_mappings()

        # Load and prepare datasets
        self.prepare_datasets()

    def get_list_position(self, input_id):
        """Find the position of an ID in the class descriptions list."""
        for index, item in enumerate(self.class_descriptions):
            if item["id"] == input_id:
                return index
        return -1  # Return -1 if the id is not found

    def setup_class_mappings(self):
        if self.class_descriptions:
            # original classes and colors
            self.class_mapping = {desc["pos"]: desc["name"] for desc in self.class_descriptions}
            # Create a color mapping for visualization
            self.class_colors = {desc["pos"]: desc["color"] for desc in self.class_descriptions}

            # Only set up merged classes if merging is enabled
            if self.merge_classes:
                # Get first new id of merge_map
                first_value = next(iter(merge_map.values()))

                # Step 1: original mapping
                self.class_mapping_merged = {
                    desc["pos"]: desc["name"]
                    for desc in self.class_descriptions
                    if desc["pos"] <= first_value - 1
                }

                self.class_colors_merged = {
                    desc["pos"]: desc["color"]
                    for desc in self.class_descriptions
                    if desc["pos"] <= first_value - 1
                }

                # Step 2: add merged classes
                id_to_desc = {desc["pos"]: desc for desc in self.class_descriptions}

                # new pos
                new_pos = first_value

                for new_id, old_ids in merge_groups.items():
                    # Take the description from the first ID in the merge group
                    representative_id = old_ids[0]
                    representative_desc = id_to_desc[representative_id]

                    self.class_mapping_merged[new_pos] = representative_desc["name"]
                    self.class_colors_merged[new_pos] = representative_desc["color"]

                    new_pos += 1
            else:
                # If not merging, use the original mapping for visualization too
                self.class_mapping_merged = self.class_mapping
                self.class_colors_merged = self.class_colors

    def parse_annotation(self, img_path, ann_path):
        """Parse XML annotation file and extract bounding boxes and class information."""
        tree = ET.parse(ann_path)
        root = tree.getroot()

        # Check if the image file exists
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            return None, None, None

        boxes = []
        class_labels = []

        for obj in root.iter("object"):
            id_elem = obj.find('id')
            label_text = int(id_elem.text) if id_elem is not None else None
            if label_text is None:
                continue

            label_id = self.get_list_position(label_text)
            if label_id == -1:
                continue

            # Only merge classes if merging is enabled
            if self.merge_classes:
                label_id = merge_map.get(label_id, label_id)

            class_labels.append(label_id)

            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            boxes.append([xmin, ymin, xmax, ymax])

        return boxes, class_labels

    def load_dataset(self):
        """Load image paths, bounding boxes and class labels from the dataset directory."""
        all_image_paths = []
        all_annotation_paths = []
        all_boxes = []
        all_classes = []

        for view in ['front', 'side']:
            img_dir = os.path.join(self.dataset_base, view, 'images')
            ann_dir = os.path.join(self.dataset_base, view, 'annotations')

            img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                ann_path = os.path.join(ann_dir, os.path.splitext(img_file)[0] + '.xml')

                if os.path.exists(ann_path):
                    all_image_paths.append(img_path)
                    all_annotation_paths.append(ann_path)

        for img_path, ann_path in zip(all_image_paths, all_annotation_paths):
            boxes, classes = self.parse_annotation(img_path, ann_path)
            all_boxes.append(boxes)
            all_classes.append(classes)
        return all_image_paths, all_boxes, all_classes

    def load_image(self, image_path):
        """Load and decode image from file path."""
        image = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
        return image

    def load_dataset_sample(self, image_path, classes, bbox):
        """Load image and format bounding boxes for KerasCV."""
        # Read Image
        image = self.load_image(image_path)

        # Format bounding boxes for KerasCV
        bounding_boxes = {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": bbox,
        }
        return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

    def dict_to_tuple(self, inputs):
        """Convert preprocessing dictionary to model input format."""
        return inputs["images"], inputs["bounding_boxes"]

    def prepare_datasets(self):
        """Prepare training and validation datasets."""
        # Load data
        image_paths, boxes, classes = self.load_dataset()

        if not image_paths:
            raise ValueError("No valid training/validation data found!")

        # Store dataset statistics for later use
        self.original_image_paths = image_paths
        self.original_boxes = boxes
        self.original_classes = classes

        # Convert to ragged tensors
        image_paths = tf.ragged.constant(image_paths)
        classes = tf.ragged.constant(classes)
        bbox = tf.ragged.constant(boxes)

        # Create dataset
        data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

        # Shuffle dataset
        if self.shuffle:
            data = data.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=False)

        if self.split == 'test':
            self.val_data = data.take(int(len(image_paths)))
            print(f"Test samples: {len(image_paths)}")
        else:
            # Determine the number of validation samples
            num_val = int(len(image_paths) * self.split_ratio)
            # Split the dataset into train and validation sets
            self.val_data = data.take(num_val)
            self.train_data = data.skip(num_val)
            print(f"Training samples: {len(image_paths) - num_val}")
            print(f"Validation samples: {num_val}")

        # Create augmentation pipelines
        if self.augmentation:
            self.setup_augmentation()

        # Setup resizer
        self.setup_resizer()

        # Prepare final datasets
        self.prepare_val_dataset()
        if self.split == 'train' or self.split == 'val':
            self.prepare_train_dataset()

    def setup_augmentation(self):
        """Set up data augmentation pipelines for training and validation."""
        # Define augmentation pipeline for training
        self.augmenter = keras.Sequential(
            layers=[
                keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
                # keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"), # does problems with bounding boxes
                keras_cv.layers.JitteredResize(target_size=self.image_size,
                                               scale_factor=(self.augmentation_scale_factor),
                                               bounding_box_format="xyxy")
            ]
        )

    def setup_resizer(self):
        self.resizer = keras_cv.layers.JitteredResize(target_size=self.image_size,
                                                      scale_factor=(self.augmentation_scale_factor),
                                                      bounding_box_format="xyxy")

    def prepare_train_dataset(self):
        """Prepare the training dataset with augmented data added to original data."""
        # Create the original dataset without augmentation
        original_ds = self.train_data.map(self.load_dataset_sample, num_parallel_calls=tf.data.AUTOTUNE)

        # Apply simple resizing to original data (but no augmentation yet)
        original_ds = original_ds.ragged_batch(self.batch_size, drop_remainder=True)
        original_ds = original_ds.map(self.resizer, num_parallel_calls=tf.data.AUTOTUNE)

        if self.augmentation and self.augmentation_copies > 0:
            # Create augmented versions of the dataset
            augmented_datasets = []

            # Create copies with augmentation
            for i in range(self.augmentation_copies):
                # Start with the original unbatched data
                aug_ds = self.train_data.map(self.load_dataset_sample, num_parallel_calls=tf.data.AUTOTUNE)

                # Apply random shuffling for each augmented copy to ensure variety
                aug_ds = aug_ds.shuffle(self.batch_size * 8)

                # Apply batching
                aug_ds = aug_ds.ragged_batch(self.batch_size, drop_remainder=True)

                # Apply augmentation
                aug_ds = aug_ds.map(self.augmenter, num_parallel_calls=tf.data.AUTOTUNE)

                # Add to our list of augmented datasets
                augmented_datasets.append(aug_ds)

            # Concatenate the original dataset with all augmented datasets
            train_ds = original_ds
            for aug_ds in augmented_datasets:
                train_ds = train_ds.concatenate(aug_ds)

            # Shuffle the combined dataset
            train_ds = train_ds.shuffle(self.batch_size * 8)
        else:
            # If no augmentation requested, just use the original dataset
            train_ds = original_ds

        # Finalize dataset preparation
        train_ds = train_ds.map(self.dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        # Report the expanded dataset size
        original_size = len(self.train_data) // self.batch_size
        expanded_size = original_size * (1 + self.augmentation_copies)
        print(f"Original training samples: {original_size * self.batch_size}")
        print(f"Expanded training samples with augmentation: ~{expanded_size * self.batch_size}")

    def prepare_val_dataset(self):
        """Prepare the validation dataset with simple resizing."""
        val_ds = self.val_data.map(self.load_dataset_sample, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.ragged_batch(self.batch_size, drop_remainder=True)
        val_ds = val_ds.map(self.resizer, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(self.dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
        self.val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    def visualize_dataset(self, dataset, rows=2, cols=2):
        inputs = next(iter(dataset.take(1)))
        if isinstance(inputs, tuple):
            images, bounding_boxes = inputs
            inputs = {"images": images, "bounding_boxes": bounding_boxes}

        images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
        visualization.plot_bounding_box_gallery(
            images,
            value_range=(0, 255),
            rows=rows,
            cols=cols,
            y_true=bounding_boxes,
            scale=5,
            font_scale=0.7,
            bounding_box_format="xyxy",
            class_mapping=self.class_mapping_merged if hasattr(self, 'class_mapping_merged') else None
        ).show()

    def get_train_dataset(self):
        """Return the prepared training dataset."""
        if self.split == 'test':
            print("Warning no train data")
        return self.train_ds

    def get_val_dataset(self):
        """Return the prepared validation dataset."""
        return self.val_ds

    def get_test_dataset(self):
        """Return the prepared validation dataset."""
        return self.val_ds

    def get_datasets(self):
        """Return both training and validation datasets."""
        if self.split == 'test':
            return self.val_ds
        else:
            return self.train_ds, self.val_ds

    def print_dataset_statistics(self):
        """
        Print statistics about the dataset:
        - Number of classes (original vs merged)
        - Number of annotations (original and augmented)
        - Number of classes annotated in datasets (original and augmented)
        """
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)

        # 1. Number of classes
        if self.merge_classes:
            num_original_classes = len(set(self.class_mapping.keys()))
            num_merged_classes = len(set(self.class_mapping_merged.keys()))
            print(f"Classes: {num_merged_classes} (merged from {num_original_classes} original classes)")
            print(f"Class merging: ENABLED")
        else:
            num_classes = len(set(self.class_mapping.keys()))
            print(f"Classes: {num_classes} (original, no merging)")
            print(f"Class merging: DISABLED")

        # 2. Count annotations in original data
        total_annotations = 0
        for boxes in self.original_boxes:
            if boxes:
                total_annotations += len(boxes)

        # Count unique classes in original data
        all_classes = []
        for class_list in self.original_classes:
            if class_list:
                all_classes.extend(class_list)

        unique_classes_original = set(all_classes)

        # 3. Count annotations after augmentation
        if self.split == 'test' or not self.augmentation:
            print(f"\nTotal annotations: {total_annotations} (no augmentation)")
        else:
            # Original annotations in training set (excluding validation)
            train_ratio = 1 - self.split_ratio
            train_annotations = int(total_annotations * train_ratio)

            # Calculate augmented annotations
            augmented_annotations = train_annotations * (1 + self.augmentation_copies)
            val_annotations = total_annotations - train_annotations

            print(f"\nOriginal annotations: {total_annotations}")
            print(f"  - Training set: {train_annotations}")
            print(f"  - Validation set: {val_annotations}")

            if self.augmentation_copies > 0:
                print(
                    f"Augmented annotations: {augmented_annotations} (with {self.augmentation_copies} additional copies per image)")

        # 4. Print class distribution in original dataset
        class_counter = collections.Counter(all_classes)

        # Sort by class ID
        sorted_classes = sorted(class_counter.items())

        print("\nClass distribution:")
        for cls_id, count in sorted_classes:
            cls_name = self.class_mapping.get(cls_id, f"Unknown ({cls_id})")
            print(f"  - {cls_name} (ID: {cls_id}): {count} annotations")

        print(f"\nTotal unique classes in dataset: {len(unique_classes_original)}")
        print("=" * 50)


# --- Example Usage ---
if __name__ == "__main__":
    # Configuration options
    BATCH_SIZE = 1
    IMAGE_SIZE = (640, 640)  # Or your preferred size
    SPLIT_RATIO = 0.1

    # Initialize the dataset loader with merged classes and expanded dataset
    train_loader = BoundingBoxDatasetLoader(
        split="test",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        split_ratio=SPLIT_RATIO,
        merge_classes=True,  # Enable/disable class merging
        augmentation=True,  # Enable augmentation
        augmentation_copies=2  # Add 2 augmented copies for each original image (3x total data)
    )

    # Print dataset statistics
    train_loader.print_dataset_statistics()

    # Get the datasets
    train_ds, val_ds = train_loader.get_datasets()

    # Or get them individually
    #train_ds = train_loader.get_train_dataset()
    #val_ds = train_loader.get_val_dataset()

    print(f"Train dataset size: {len(train_ds) * train_loader.batch_size} samples")
    print(f"Validation dataset size: {len(val_ds) * train_loader.batch_size} samples")
    print(f"Num classes: {len(train_loader.class_mapping_merged)}")


    # Initialize the dataset loader with original unmerged classes
    test_loader = BoundingBoxDatasetLoader(
        split="test",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        split_ratio=SPLIT_RATIO,
        merge_classes=True,  # Disable class merging
        augmentation=True  # No augmentation for test data
    )

    # Print test dataset statistics
    test_loader.print_dataset_statistics()

    # Get the dataset
    test_ds = test_loader.get_datasets()

    # Or get them individually
    # test_ds = test_loader.get_test_dataset()

    print(f"Test dataset size: {len(test_ds) * test_loader.batch_size} samples")

    print("Visualizing dataset with merged classes:")
    train_loader.visualize_dataset(train_ds, rows=2, cols=2)
    train_loader.visualize_dataset(val_ds, rows=2, cols=2)

    print("Visualizing dataset with original (unmerged) classes:")
    test_loader.visualize_dataset(test_ds, rows=2, cols=2)
