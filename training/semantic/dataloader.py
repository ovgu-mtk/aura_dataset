import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import aura_dataset.utils.classes as cls

images_path = "../../dataset/semantic/train_validate/images"
annotations_path = "../../dataset/semantic/train_validate/annotations"

# map colors to merge classes
merge_map = {
            3: 2,   #  shared bicycle-/pedestrian path -> bicycle lane
            8: 4,   # parking_space -> car_lane
            11: 12, # centre line -> lane boundary
            16: 15,  # cyclist without bicycle -> cyclist with bicycle
            17: 15,  # motorcyclist with motorcycle -> cyclist with bicycle
            18: 15,  # motorcyclist without motorcycle -> cyclist with bicycle
        }


class DataLoader(Sequence):
    def __init__(self,
                 images_path=images_path,
                 annotations_path=annotations_path,
                 batch_size=4,
                 img_size=(512, 1024),
                 shuffle=True,
                 split='train',
                 val_split=0.1,
                 num_classes=26,
                 merge_classes=False):

        self.images_path = images_path
        self.annotations_path = annotations_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.split = split
        self.num_classes = num_classes
        self.merge_classes = merge_classes

        self.image_filenames = sorted(os.listdir(images_path))
        self.mask_filenames = sorted(os.listdir(annotations_path))

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

    train_gen = DataLoader(batch_size=batch_size,
                 img_size=img_size,
                 shuffle=False,
                 split='train',
                 val_split=0.1,
                 num_classes=26,
                 merge_classes=True)

    val_gen = DataLoader(batch_size=batch_size,
                 img_size=img_size,
                 shuffle=False,
                 split='val',
                 val_split=0.1,
                 num_classes=26,
                 merge_classes=True)


    print(len(train_gen))
    print(len(val_gen))

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