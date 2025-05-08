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
base_dataset_path = str(find_root(__file__ , "aura_dataset")) + "/dataset/semantic/train_validate/"


# Define merge groups: new_id -> [old_ids...]
merge_groups = {
    1: [2],                    # pedestrian path
    2: [3, 7],                 # car lane
    3: [4],                    # sealed free space
    4: [5],                    # unsealed path
    5: [6],                    # horizontal vegetation
    6: [8],                    # railway track
    7: [9],                    # curb stone
    8: [10,11],                # lane stripes
    9: [12],                   # vertical barrier
    10: [13],                  # vehicle
    11: [14,15,16,17],         # cyclist
    12: [18],                  # pedestrian
    13: [19],                  # traffic sign
    14: [20],                  # traffic light
    15: [21],                  # vertical vegetation
    16: [22],                  # sky
    17: [23],                  # miscellanous static object
    18: [24],                  # miscellanous dynamic object
}

# Flatten to a merge_map
merge_map = {old_id: new_id for new_id, old_ids in merge_groups.items() for old_id in old_ids}


class SemanticDataLoader(Sequence):
    def __init__(self, base_path=base_dataset_path,
                 batch_size=4,
                 img_size=(1086, 2046),
                 shuffle=True,
                 split='train',
                 val_split=0.1,
                 merge_classes=True,
                 class_descriptions=cls.class_descriptions_semantic):

        self.images_path = base_path + "images"
        self.annotations_path = base_path + "annotations"
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.split = split
        self.merge_classes = merge_classes

        if self.merge_classes:
            self.number_of_classes = self.count_merged_classes(class_descriptions, merge_groups)
        else:
            self.number_of_classes = len(class_descriptions)

        # Set up class descriptions if provided, otherwise use an empty list
        self.class_descriptions = class_descriptions if class_descriptions else []

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

    def count_merged_classes(self, original_classes, merge_groups):
        merged_ids = set()  # all old_ids that are being merged
        for old_ids in merge_groups.values():
            merged_ids.update(old_ids)

        destination_ids = set(merge_groups.keys())  # new class IDs

        untouched_ids = [
            class_id for class_id in original_classes.keys()
            if class_id not in merged_ids and class_id not in destination_ids
        ]

        total_classes = len(destination_ids) + len(untouched_ids)
        return total_classes


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
                for old_class, new_class in merge_map.items():
                    mask[mask == old_class] = new_class

            masks.append(mask)
            images.append(img)

        return np.array(images), np.array(masks)

    def visualize_results(self, model, images_path, annotations_path, img_size=(512, 1024)):
        for input_image, ground_truth in zip(os.listdir(images_path), os.listdir(annotations_path)):
            img = load_img(os.path.join(images_path, input_image), target_size=img_size)
            img_array = img_to_array(img) / 255.0
            img_input = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_input)[0]  # shape: (512, 1024, 26)
            pred_mask = np.argmax(pred, axis=-1)  # shape: (512, 1024), values in [0, 25]

            mask = load_img(os.path.join(annotations_path, ground_truth), target_size=img_size, color_mode='grayscale')
            mask_array = np.squeeze(img_to_array(mask).astype(np.uint8))

            pred_color = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            gt_color = np.zeros_like(pred_color)

            for class_idx, color in cls.class_descriptions_semantic.items():
                pred_color[pred_mask == class_idx] = color
                gt_color[mask_array == class_idx] = color

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
                 merge_classes=True)

    val_gen = SemanticDataLoader(batch_size=batch_size,
                 img_size=img_size,
                 shuffle=False,
                 split='val',
                 val_split=0.1,
                 merge_classes=False)


    print(len(train_gen))
    print(len(val_gen))
    print(train_gen.number_of_classes)
    print(val_gen.number_of_classes)


    for i in range(20):
        # use first batch of training data for plot
        X_batch, y_batch = train_gen[i]

        # Load RGB and grayscale images using PIL
        img = X_batch[0]
        gray = y_batch[0]

        # Convert images to numpy arrays
        img_display = img if np.max(img) <= 1.0 else img / 255.0
        gray = np.array(gray)

        rows, cols, _ = gray.shape
        np_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        # Apply the color map
        for i in range(rows):
            for j in range(cols):
                value = gray[i, j, 0]  # Use the first channel since the mask is (H, W, 1)
                color = cls.class_descriptions_semantic[int(value)]  # Get the color based on the integer value
                np_image[i, j] = color

        # Display the original and semantic images using matplotlib
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        axs[0].imshow(img_display)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(np_image)
        axs[1].set_title('Semantic Segmentation')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

        # Pause until a key is pressed
        input("Press any key to continue...")