import keras_cv

def create_yolo_model(dataloader, backbone, fpn_depth=3):
    # Create YOLO detector with pre-trained backbone
    model = keras_cv.models.YOLOV8Detector(num_classes=len(dataloader.class_mapping_merged), bounding_box_format="xyxy",
                                           backbone=keras_cv.models.YOLOV8Backbone.from_preset(backbone),
                                           fpn_depth=fpn_depth)

    return model

