import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import classes as cls


# Paths to dataset
images_path = "../dataset/semantic/train_validate/images/"
annotations_path = "../dataset/semantic/train_validate/annotations/"

# Process images
for image_name in os.listdir(images_path):
    if image_name.endswith(".png"):
        # Load RGB and grayscale images using PIL
        img = Image.open(os.path.join(images_path, image_name)).convert("RGB")
        gray = Image.open(os.path.join(annotations_path, image_name)).convert("L")

        # Resize images
        img = img.resize((682, 362), Image.BILINEAR)
        gray = gray.resize((682, 362), Image.NEAREST)

        # Convert images to numpy arrays
        img = np.array(img)
        gray = np.array(gray)

        rows, cols = gray.shape
        np_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        # Apply the color map
        for i in range(rows):
            for j in range(cols):
                value = gray[i, j]
                color = cls.class_descriptions_semantic[value]
                np_image[i, j] = color

        # Display the original and semantic images using matplotlib
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        axs[0].imshow(img)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(np_image)
        axs[1].set_title('Semantic Segmentation')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

        # Pause until a key is pressed
        input("Press any key to continue...")