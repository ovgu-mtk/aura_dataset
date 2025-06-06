import os
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
import random


def get_file_datetime(file_path):
    try:
        mtime = os.path.getmtime(file_path)
        m_datetime = datetime.fromtimestamp(mtime)
        return m_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    except Exception as e:
        print(f"Error getting metadata for {file_path}: {e}")
        return None

def update_xml_filename(xml_path, new_image_name):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename_element = root.find('filename')
        if filename_element is not None:
            filename_element.text = new_image_name
        tree.write(xml_path)
        return True
    except Exception as e:
        print(f"Error updating XML file {xml_path}: {e}")
        return False

def generate_unique_filename(base_filename, output_folder):
    name, ext = os.path.splitext(base_filename)
    unique_name = base_filename
    counter = 1
    while os.path.exists(os.path.join(output_folder, unique_name)):
        unique_name = f"{name}_{counter}{ext}"
        counter += 1
    return unique_name

def process_files(image_folder, annotation_folder, base_output_folder, test_split_ratio=0.1):
    # List image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)  # Shuffle randomly

    # Calculate split
    split_index = int(len(image_files) * (1 - test_split_ratio))
    train_validate_files = image_files[:split_index]
    test_files = image_files[split_index:]

    # Create output folders
    folders = ['train_validate', 'test']
    for subset in folders:
        os.makedirs(os.path.join(base_output_folder, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_output_folder, subset, 'annotations'), exist_ok=True)

    all_files = {
        'train_validate': train_validate_files,
        'test': test_files
    }

    stats = {'processed': 0, 'no_ann': [], 'no_meta': []}

    for subset, file_list in all_files.items():
        out_img_dir = os.path.join(base_output_folder, subset, 'images')
        out_ann_dir = os.path.join(base_output_folder, subset, 'annotations')

        for image_file in file_list:
            base_name = os.path.splitext(image_file)[0].replace("_original", "")
            ann_file = f"{base_name}_BB.xml"
            ann_path = os.path.join(annotation_folder, ann_file)

            if not os.path.exists(ann_path):
                print(f"Warning: No annotation file for {image_file}")
                stats['no_ann'].append(image_file)
                continue

            image_path = os.path.join(image_folder, image_file)
            datetime_str = get_file_datetime(image_path)
            if not datetime_str:
                print(f"Error: Could not retrieve metadata for {image_file}")
                stats['no_meta'].append(image_file)
                continue

            img_name = generate_unique_filename(f"{datetime_str}.jpg", out_img_dir)
            ann_name = f"{os.path.splitext(img_name)[0]}.xml"

            # Copy files
            shutil.copy2(image_path, os.path.join(out_img_dir, img_name))
            shutil.copy2(ann_path, os.path.join(out_ann_dir, ann_name))

            # Update XML
            if update_xml_filename(os.path.join(out_ann_dir, ann_name), img_name):
                print(f"[{subset}] Processed {img_name}")
            else:
                print(f"[{subset}] Failed to update XML for {img_name}")

            stats['processed'] += 1

    return stats

if __name__ == "__main__":
    # Config
    image_folder = "/media/afius/Data1/aura_dataset/bb/original/front/images/"
    annotation_folder = "/media/afius/Data1/aura_dataset/bb/original/front/annotations/"
    base_output_folder = "/home/afius/Downloads/aura_dataset/bb/modified/front/"
    test_split_ratio = 0.1  # 10% for test

    stats = process_files(image_folder, annotation_folder, base_output_folder, test_split_ratio)

    print("\n" + "=" * 50)
    print(f"PROCESSING SUMMARY:")
    print(f"Processed images: {stats['processed']}")
    print(f"No annotation files: {len(stats['no_ann'])}")
    print(f"No metadata: {len(stats['no_meta'])}")
