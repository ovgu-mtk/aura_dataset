import os
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET
import classes as cls

# Paths to dataset
images_path = "../dataset/bb/train_validate/side/images/"
annotations_path = "../dataset/bb/train_validate/side/annotations/"

# Function to get color and name for the given class id
def get_class_info(class_id, class_descriptions):
    for class_info in class_descriptions:
        if class_info['id'] == class_id:
            return class_info['name'], class_info['color']
    return None, None

# Get list of image and annotation files
image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
annotation_files = [f for f in os.listdir(annotations_path) if f.endswith('.xml')]

# Sort files to ensure matching pairs
image_files.sort()
annotation_files.sort()

# Font for annotation text (Pillow uses system fonts, fallback if not available)
try:
    font = ImageFont.truetype("arial.ttf", size=14)
except IOError:
    font = ImageFont.load_default()

# Enumerate images
for i, (image_file, annotation_file) in enumerate(zip(image_files, annotation_files)):
    image_path = os.path.join(images_path, image_file)
    annotation_path = os.path.join(annotations_path, annotation_file)

    # Load image using PIL
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Parse XML
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Iterate through all objects in the XML file
    for obj in root.findall('object'):
        class_id = int(obj.find('id').text)
        name, color = get_class_info(class_id, cls.class_descriptions_bb)

        if name is None or color is None:
            print(f"Warning: Unknown class ID {class_id} in {annotation_file}")
            continue  # skip this object

        # Convert color to tuple
        if isinstance(color, list):
            color = tuple(color)


        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Draw bounding box and class label
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
        draw.text((xmin, ymin - 15), name, fill=color, font=font)

    # Show the image (will open in default viewer)
    image.show()

    # Pause until a key is pressed
    input("Press any key to continue...")