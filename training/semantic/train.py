import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from training.semantic.dataloader import SemanticDataLoader
from training.semantic.model import compile_model, load_pretrained_hrnet


def parse_args():
    parser = argparse.ArgumentParser(description="Train semantic segmentation model.")
    parser.add_argument('--gpu', type=int, choices=range(0, 4), default=0, help='GPU index to use (0-3). Default: 0')
    parser.add_argument('--model_index', type=int, choices=range(0, 4), default=0,
                        help='Model index to name the saved model. Default: 0')
    parser.add_argument('--img_height', type=int, default=1086, help='Image height. Default: 1086')
    parser.add_argument('--img_width', type=int, default=2046, help='Image width. Default: 2046')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs. Default: 300')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size. Default: 1')
    parser.add_argument('--loss', type=str, default='ohem_loss', help='Loss function. Default: ohem_loss')
    parser.add_argument('--num_classes', type=int, choices=[13,19,25], default=25,
                        help='use all classes or a subdivision [13,19,25]. Default: 25')
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

    # Set GPU
    set_gpu(args.gpu)

    # Save path
    model_name = f"{args.model_index}_model"
    save_model_path = f"model/{model_name}.keras"

    # Load datasets
    train_loader = SemanticDataLoader(batch_size=args.batch_size,
                                      img_size=(args.img_height, args.img_width),
                                      split='train',
                                      num_classes=args.num_classes)

    val_loader = SemanticDataLoader(batch_size=args.batch_size,
                                    img_size=(args.img_height, args.img_width),
                                    split='val',
                                    num_classes=args.num_classes)


    # Load model
    model = load_pretrained_hrnet(input_shape=(args.img_height, args.img_width, 3), num_classes=train_loader.number_of_classes)


    # Compile Model
    compile_model(model, loss=args.loss)
    model.summary()

    # Print all arguments
    print("Training configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Callbacks
    checkpoint = ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    reduce_plateau = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    callbacks = [early_stopping, reduce_plateau, checkpoint]

    # Train model
    model.fit(train_loader, validation_data=val_loader, epochs=args.epochs, callbacks=callbacks)