import os
import cv2
import xml.etree.ElementTree as ET
import classes as cls


# Function to get color and name for the given class id
def get_class_info(class_id, class_descriptions):
    for class_info in class_descriptions:
        if class_info['id'] == class_id:
            return class_info['name'], class_info['color']
    return None, None


# Define a function to map object classes to colors
def get_color_for_class(object_class):
    class_colors = {
        "trafficsign_Parken": (0, 255, 0),  # Green
        "adult without baby buggy or mobility support": (255, 0, 0),  # Blue
        "car": (0, 0, 255),  # Red
        "bicycle without cyclist": (255, 255, 0)  # Cyan
    }
    return class_colors.get(object_class, (255, 255, 255))  # Default to white if class not found

image_folder = "D:/aura_dataset/bounding_boxes/front/images/"
annotation_folder = "D:/aura_dataset/bounding_boxes/front/annotations/"

# Create output folders if they do not exist
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
if not os.path.exists(annotation_folder):
    os.makedirs(annotation_folder)

# Get list of image and annotation files
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith('.xml')]

# Sort files to ensure matching pairs
image_files.sort()
annotation_files.sort()


# enumerate images
for i, (image_file, annotation_file) in enumerate(zip(image_files, annotation_files)):

    # Read the image
    image = cv2.imread(image_folder + image_file)

    # Read the xml file
    tree = ET.parse(annotation_folder + annotation_file)
    root = tree.getroot()

    # Iterate through all objects in the XML file
    for obj in root.findall('object'):
        # object_class = obj.find('name').text
        class_id = int(obj.find('id').text)
        name, color = get_class_info(class_id, cls.class_descriptions_bb)
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Get the color for the class
        # color = get_color_for_class(object_class)

        # Draw the bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        # Put the class name near the bounding box
        cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save or display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


