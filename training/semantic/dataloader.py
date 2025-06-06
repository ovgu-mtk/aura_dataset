import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import utils.classes as cls


def find_root(start_path, target_name):
    current_path = Path(start_path).resolve()

    # Traverse up the directory tree
    for parent in current_path.parents:
        if parent.name == target_name:
            return parent

    return None  # Return None if no match is found


# paths
train_val_dataset_path = str(find_root(__file__, "aura_dataset")) + "/dataset/semantic/train_validate/"
test_dataset_path = str(find_root(__file__, "aura_dataset")) + "/dataset/semantic/test/"

# Define merge groups: new_id -> [old_ids...]
merge_groups_19 = {
    0: [1],  # bicycle lane
    1: [2, 3],  # pedestrian path
    2: [4, 8],  # car lane/ parking space
    3: [5],  # sealed free space
    4: [6],  # unsealed path
    5: [7],  # horizontal vegetation
    6: [9],  # railway track
    7: [10],  # curb stone
    8: [11, 12],  # lane stripes
    9: [13],  # vertical barrier
    10: [14],  # vehicle
    11: [15, 16, 17, 18],  # cyclist
    12: [19],  # pedestrian
    13: [20],  # traffic sign
    14: [21],  # traffic light
    15: [22],  # vertical vegetation
    16: [23],  # sky
    17: [24],  # miscellanous static object
    18: [25],  # miscellanous dynamic object
}


# Define merge groups: new_id -> [old_ids...]
merge_groups_13 = {
    0: [1, 2, 3, 5, 6],  # pedestrian and bicycle lane
    1: [4, 8],  # car lane/ parking space
    2: [7, 22],  # vegetation
    3: [9],  # railway track
    4: [10],  # curb stone
    5: [11, 12],  # lane stripes
    6: [13],  # vertical barrier
    7: [14],  # vehicle
    8: [15, 16, 17, 18],  # cyclist
    9: [19],  # pedestrian
    10: [20, 21],  # traffic sign
    11: [23],  # sky
    12: [24, 25],  # miscellanous object
}



# Classes to exclude (when not merging)
excluded_classes = [0]  # ego-vehicle


class SemanticDataLoader(Sequence):
    def __init__(self,
                 batch_size=4,
                 img_size=(1086, 2046),
                 shuffle=True,
                 split='train',
                 val_split=0.1,
                 num_classes=25,
                 class_descriptions=cls.class_descriptions_semantic):

        if split == 'train' or split == 'val':
            base_path = train_val_dataset_path
        elif split == 'test':
            base_path = test_dataset_path
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.images_path = base_path + "images"
        self.annotations_path = base_path + "annotations"
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.split = split

        if num_classes == 25:
            self.merge_classes = False
        elif num_classes == 19:
            self.merge_classes = True
            self.merge_groups = merge_groups_19
        elif num_classes == 13:
            self.merge_classes = True
            self.merge_groups = merge_groups_13
        else:
            raise ValueError("num classe must be '25', '19', or '13'")



        if self.merge_classes:
            # Flatten to a merge_map
            self.merge_map = {old_id: new_id for new_id, old_ids in self.merge_groups.items() for old_id in old_ids}
            # Set up class descriptions if provided, otherwise use an empty list
            self.class_descriptions = class_descriptions if class_descriptions else []
            self.number_of_classes = self.count_merged_classes(class_descriptions)
            # Create filtered class descriptions for merged classes
            self.filtered_class_descriptions = self.create_merged_class_descriptions(class_descriptions)
        else:
            # Remove excluded classes and create class remapping
            self.filtered_class_descriptions = {k: v for k, v in class_descriptions.items() if
                                                k not in excluded_classes}
            self.number_of_classes = len(self.filtered_class_descriptions)
            # Create remapping from original class IDs to new sequential IDs (0, 1, 2, ...)
            self.class_remap = {old_id: new_id for new_id, old_id in
                                enumerate(sorted(self.filtered_class_descriptions.keys()))}

        self.image_filenames = sorted(os.listdir(self.images_path))
        self.mask_filenames = sorted(os.listdir(self.annotations_path))

        assert len(self.image_filenames) == len(
            self.mask_filenames), "Number of input images and masks must be the same"

        indices = np.arange(len(self.image_filenames))
        train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42)

        if split == 'train':
            self.indices = train_idx
        elif split == 'val':
            self.indices = val_idx
        elif split == 'test':
            self.indices = indices
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_filenames = [self.image_filenames[i] for i in indices]
        batch_mask_filenames = [self.mask_filenames[i] for i in indices]

        images, masks = self.__data_generation(batch_image_filenames, batch_mask_filenames)

        return images, masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def count_merged_classes(self, original_classes):
        merged_ids = set()  # all old_ids that are being merged
        for old_ids in self.merge_groups.values():
            merged_ids.update(old_ids)

        destination_ids = set(self.merge_groups.keys())  # new class IDs

        untouched_ids = [
            class_id for class_id in original_classes.keys()
            if class_id not in merged_ids and class_id not in destination_ids
        ]

        total_classes = len(destination_ids) + len(untouched_ids)
        return total_classes

    def create_merged_class_descriptions(self, original_classes):
        """Create class descriptions for merged classes"""
        merged_descriptions = {}

        # Add merged classes (use the color of the first class in each group)
        for new_id, old_ids in self.merge_groups.items():
            if old_ids:  # Make sure the list is not empty
                merged_descriptions[new_id] = original_classes[old_ids[0]]

        # Add untouched classes
        merged_ids = set()
        for old_ids in self.merge_groups.values():
            merged_ids.update(old_ids)
        destination_ids = set(self.merge_groups.keys())

        for class_id, color in original_classes.items():
            if class_id not in merged_ids and class_id not in destination_ids:
                merged_descriptions[class_id] = color

        return merged_descriptions

    def __data_generation(self, batch_image_filenames, batch_mask_filenames):
        images = []
        masks = []

        for img_filename, mask_filename in zip(batch_image_filenames, batch_mask_filenames):
            img_path = os.path.join(self.images_path, img_filename)
            mask_path = os.path.join(self.annotations_path, mask_filename)

            img = load_img(img_path, target_size=self.img_size, color_mode='rgb')
            img = img_to_array(img) / 255.0

            mask = load_img(mask_path, target_size=self.img_size, color_mode='grayscale')
            mask = np.expand_dims(mask, axis=-1)  # Make it (H, W, 1)
            mask = np.copy(mask)

            if self.merge_classes:
                # Apply merging
                for old_class, new_class in self.merge_map.items():
                    mask[mask == old_class] = new_class
            else:
                # Remove excluded classes and remap remaining classes to sequential IDs
                for excluded_class in excluded_classes:
                    mask[mask == excluded_class] = 255  # Set to ignore value (or handle as you prefer)

                # Remap remaining classes to sequential IDs (0, 1, 2, ...)
                temp_mask = np.full_like(mask, 255)  # Initialize with ignore value
                for old_id, new_id in self.class_remap.items():
                    temp_mask[mask == old_id] = new_id
                mask = temp_mask

            masks.append(mask)
            images.append(img)

        return np.array(images), np.array(masks)

    def visualize_results(self, model, img_size=(1086, 2046)):
        for input_image, ground_truth in zip(os.listdir(self.images_path), os.listdir(self.annotations_path)):
            img = load_img(os.path.join(self.images_path, input_image), target_size=img_size)
            img_array = img_to_array(img) / 255.0
            img_input = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_input)[0]  # shape: (1086, 2046, num_classes)
            pred_mask = np.argmax(pred, axis=-1)  # shape: (1086, 2046, values in [0, num_classes-1]

            # Load ground truth mask
            mask = load_img(os.path.join(self.annotations_path, ground_truth), target_size=img_size,
                            color_mode='grayscale')
            mask_array = np.squeeze(img_to_array(mask).astype(np.uint8))

            # Apply the same preprocessing as in __data_generation
            mask_array = np.expand_dims(mask_array, axis=-1)  # Make it (H, W, 1)
            mask_array = np.copy(mask_array)

            if self.merge_classes:
                # Apply merging to ground truth mask
                for old_class, new_class in self.merge_map.items():
                    mask_array[mask_array == old_class] = new_class
            else:
                # Remove excluded classes and remap remaining classes to sequential IDs
                for excluded_class in excluded_classes:
                    mask_array[mask_array == excluded_class] = 255  # Set to ignore value

                # Remap remaining classes to sequential IDs (0, 1, 2, ...)
                temp_mask = np.full_like(mask_array, 255)  # Initialize with ignore value
                for old_id, new_id in self.class_remap.items():
                    temp_mask[mask_array == old_id] = new_id
                mask_array = temp_mask

            # Squeeze back to (H, W) for visualization
            mask_array = np.squeeze(mask_array)

            pred_color = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            gt_color = np.zeros_like(pred_color)

            # Use the appropriate class descriptions based on merge_classes setting
            class_descriptions = self.filtered_class_descriptions

            # Apply colors to ground truth
            if self.merge_classes:
                # For merged classes, use the class IDs directly
                for class_idx, color in class_descriptions.items():
                    gt_color[mask_array == class_idx] = color
            else:
                # For non-merged classes, map from sequential IDs (0,1,2...) back to colors
                # Only color pixels that are not ignored (value 255)
                valid_mask = mask_array != 255
                for sequential_id, (original_class_id, color) in enumerate(class_descriptions.items()):
                    gt_color[(mask_array == sequential_id) & valid_mask] = color

            # Apply colors to prediction - need to map from model output indices to class IDs
            if self.merge_classes:
                # For merged classes, the model outputs indices 0 to num_classes-1
                # We need to map these back to the merged class IDs
                sorted_class_ids = sorted(class_descriptions.keys())
                for model_idx, class_id in enumerate(sorted_class_ids):
                    if class_id in class_descriptions:
                        color = class_descriptions[class_id]
                        pred_color[pred_mask == model_idx] = color
            else:
                # For non-merged classes, map from sequential model output to original class IDs
                for model_idx, (original_class_id, color) in enumerate(class_descriptions.items()):
                    pred_color[pred_mask == model_idx] = color

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].imshow(img)
            axs[0].set_title('Original Image')
            axs[0].axis('off')

            axs[1].imshow(gt_color)
            axs[1].set_title('Ground Truth')
            axs[1].axis('off')

            axs[2].imshow(pred_color)
            axs[2].set_title('Prediction')
            axs[2].axis('off')

            plt.tight_layout()
            plt.show()


# Example Usage
if __name__ == "__main__":
    batch_size = 1
    img_size = (1024, 2048)

    train_gen = SemanticDataLoader(batch_size=batch_size,
                                   img_size=img_size,
                                   shuffle=False,
                                   split='train',
                                   val_split=0.1,
                                   num_classes=25) # 25, 19, 13

    val_gen = SemanticDataLoader(batch_size=batch_size,
                                 img_size=img_size,
                                 shuffle=False,
                                 split='test',
                                 val_split=0.1,
                                 num_classes=25)  # This will now exclude class 0

    print(f"Train generator length: {len(train_gen)}")
    print(f"Val generator length: {len(val_gen)}")
    print(f"Train classes (merged): {train_gen.number_of_classes}")
    print(f"Val classes (original minus excluded): {val_gen.number_of_classes}")
    #print(f"Val class mapping: {val_gen.class_remap}")

    for i in range(10):
        # use first batch of training data for plot
        X_batch, y_batch = val_gen[i]  # Use val_gen to see the effect of class removal

        # Load RGB and grayscale images using PIL
        img = X_batch[0]
        gray = y_batch[0]

        # Convert images to numpy arrays
        img_display = img if np.max(img) <= 1.0 else img / 255.0
        gray = np.array(gray)

        rows, cols, _ = gray.shape
        np_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        # Apply the color map using filtered class descriptions
        for i in range(rows):
            for j in range(cols):
                value = gray[i, j, 0]  # Use the first channel since the mask is (H, W, 1)
                if value != 255:  # Ignore pixels with value 255 (excluded classes)
                    if not val_gen.merge_classes:
                        # For non-merged classes, we need to map back from remapped IDs to original colors
                        # Find original class ID from remapped ID
                        original_id = None
                        for orig_id, remap_id in val_gen.class_remap.items():
                            if remap_id == int(value):
                                original_id = orig_id
                                break
                        if original_id is not None:
                            color = val_gen.filtered_class_descriptions[original_id]
                            np_image[i, j] = color
                    else:
                        # For merged classes, use the value directly
                        if int(value) in val_gen.filtered_class_descriptions:
                            color = val_gen.filtered_class_descriptions[int(value)]
                            np_image[i, j] = color

        # Display the original and semantic images using matplotlib
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        axs[0].imshow(img_display)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(np_image)
        axs[1].set_title('Semantic Segmentation (Class 0 Excluded)')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

        # Pause until a key is pressed
        input("Press any key to continue...")