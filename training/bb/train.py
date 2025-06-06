import os
import argparse
import tensorflow as tf
from tensorflow import keras
from training.bb.dataloader import BoundingBoxDatasetLoader
from training.bb.model import create_yolo_model
from utils.metrics import EvaluateCOCOMetricsCallback

def parse_args():
    parser = argparse.ArgumentParser(description="Train bounding box model.")
    parser.add_argument('--gpu', type=int, choices=range(0, 4), default=3, help='GPU index to use (0-3). Default: 0')
    parser.add_argument('--model_index', type=int, choices=range(0, 4), default=3,
                        help='Model index to name the saved model. Default: 0')
    parser.add_argument('--model_backbone', type=str, default='yolo_v8_xs_backbone_coco',
                        help='Model backbone. Options: yolo_v8_xs_backbone_coco, yolo_v8_s_backbone_coco, '
                             'yolo_v8_m_backbone_coco, yolo_v8_l_backbone_coco, yolo_v8_xl_backbone_coco. '
                             'Default: yolo_v8_s_backbone_coco')
    parser.add_argument('--img_height', type=int, default=1086, help='Image height. Default: 1086')
    parser.add_argument('--img_width', type=int, default=2046, help='Image width. Default: 2046')
    parser.add_argument('--use_augmentation', type=str, choices=['true', 'false'], default='true',
                        help='Activate/Deactivate augmentation. Default: true')
    parser.add_argument('--augmentation_copies', type=int, default=2, help='Number of augmentation copies to increase training data. Default: True')
    parser.add_argument('--augmentation_scale_factor', type=float, nargs=2, default=(0.8, 1.25),
                        help='Modify augmentation scale factor for scale the input image. Default: (0.8, 1.25)')
    parser.add_argument('--merge_classes', type=str, choices=['true', 'false'], default='true',
                        help='Activate/Deactivate merge classes. Default: true')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs. Default: 300')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size. Default: 1')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split to train on. Default: train')
    parser.add_argument('--split_ratio', type=float, default=0.1, help='Split ratio to split dataset in training/validation'
                                           'e.g. 0.1=# 90% training, 10% validation: 1. Default: 0.1')
    return parser.parse_args()


def set_gpu(gpu_index):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
        except Exception as e:
            print(f"Failed to use GPU {gpu_index}, falling back to GPU 0. Error: {e}")
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)



if __name__ == "__main__":

    args = parse_args()

    # Convert string values to boolean
    args.use_augmentation = args.use_augmentation.lower() == 'true'
    args.merge_classes = args.merge_classes.lower() == 'true'

    # Set GPU
    set_gpu(args.gpu)

    # Save path
    model_name = f"{args.model_index}_model"
    save_model_path = f"model/{model_name}.keras"


    # Load datasets
    # Initialize the dataset loader with merged classes and expanded dataset
    train_loader = BoundingBoxDatasetLoader(
        batch_size=args.batch_size,
        image_size=(args.img_height, args.img_width),
        split_ratio=args.split_ratio,
        split=args.split,
        augmentation=args.use_augmentation,  # Enable augmentation
        augmentation_copies=args.augmentation_copies,
        augmentation_scale_factor=args.augmentation_scale_factor,
        merge_classes = args.merge_classes,  # Enable/disable class merging
    )

    # Print dataset statistics
    train_loader.print_dataset_statistics()
    train_ds, val_ds = train_loader.get_datasets()

    # Create Model
    model = create_yolo_model(dataloader=train_loader, backbone=args.model_backbone)

    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, global_clipnorm=10.0)
    model.compile(optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou")

    # print model architecture
    #model.summary()

    # Print all arguments
    print(f"Training model: {model_name}.keras with configuration: "
          f"\n  number of classes: {len(train_loader.class_mapping_merged)}")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    """
    ## Training
    """
    # Add early stopping callback to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Add learning rate scheduler to improve training
    reduce_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Create a ModelCheckpoint callback for regular checkpoints
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_loss', mode='min')

    # coco metric
    coco_metric = EvaluateCOCOMetricsCallback(val_ds),

    callbacks = [coco_metric, early_stopping, reduce_plateau, checkpoint]

    # Train the model
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)