from struct import unpack
import os
import shutil
from tqdm import tqdm

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while (True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]
            if len(data) == 0:
                raise TypeError("issue reading jpeg file")


def check_and_delete_corrupted_images(image_folder, annotation_folder=None, delete=False, backup=False):
    """
    Check for corrupted JPEG images in the specified folder and optionally handle their annotations.

    Args:
    - image_folder (str): Path to the folder containing images
    - annotation_folder (str, optional): Path to the folder containing corresponding XML annotations
    - delete (bool): If True, delete corrupted images and their annotations
    - backup (bool): If True, move corrupted images and annotations to backup folders

    Returns:
    - list of corrupted image filenames
    """
    # Ensure folder paths are absolute
    image_folder = os.path.abspath(image_folder)

    # Create backup folders if backup is True
    if backup:
        image_backup_folder = os.path.join(image_folder, 'corrupted_image_backup')
        os.makedirs(image_backup_folder, exist_ok=True)

        if annotation_folder:
            annotation_backup_folder = os.path.join(annotation_folder, 'corrupted_annotation_backup')
            os.makedirs(annotation_backup_folder, exist_ok=True)

    # Get list of image paths
    image_paths = os.listdir(image_folder)

    # List to store corrupted image names
    corrupted_jpegs = []

    # Check each image
    for img_path in tqdm(image_paths):
        full_image_path = os.path.join(image_folder, img_path)

        # Skip if it's not a file
        if not os.path.isfile(full_image_path):
            continue

        try:
            image = JPEG(full_image_path)
            image.decode()
        except Exception:
            corrupted_jpegs.append(img_path)

            # Handle image deletion or backup
            if delete:
                os.remove(full_image_path)
                print(f"Deleted corrupted image: {img_path}")
            elif backup:
                backup_path = os.path.join(image_backup_folder, img_path)
                shutil.move(full_image_path, backup_path)
                print(f"Moved corrupted image to backup: {img_path}")

            # Handle corresponding annotation file
            if annotation_folder:
                # Construct annotation filename (assuming .xml extension)
                annotation_filename = os.path.splitext(img_path)[0] + '.xml'
                full_annotation_path = os.path.join(annotation_folder, annotation_filename)

                # Check if annotation file exists
                if os.path.exists(full_annotation_path):
                    if delete:
                        os.remove(full_annotation_path)
                        print(f"Deleted corresponding annotation: {annotation_filename}")
                    elif backup:
                        backup_annotation_path = os.path.join(annotation_backup_folder, annotation_filename)
                        shutil.move(full_annotation_path, backup_annotation_path)
                        print(f"Moved corresponding annotation to backup: {annotation_filename}")

    # Print summary
    print(f"\nFound {len(corrupted_jpegs)} corrupted images:")
    for img in corrupted_jpegs:
        print(img)

    return corrupted_jpegs


# Example usage
if __name__ == "__main__":
    image_folder = "/media/afius/Data1/aura_dataset/bb/original/front/images/"
    annotation_folder = "/media/afius/Data1/aura_dataset/bb/original/front/annotations/"

    # Option 1: Just list corrupted images
    # check_and_delete_corrupted_images(image_folder, annotation_folder)

    # Option 2: Delete corrupted images and their annotations
    check_and_delete_corrupted_images(image_folder, annotation_folder, delete=True)

    # Option 3: Move corrupted images and annotations to backup folders
    # check_and_delete_corrupted_images(image_folder, annotation_folder, backup=True)
