import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from dataloader import DataLoader
from tensorflow_model import compile_model, FastSCNN, create_unet_model, create_small_unet_pretrained, DeepLabV3Plus, load_pretrained_hrnet, simple_unet


# set cuda device 0-3
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Force TensorFlow to use GPU 1 if exist
        tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[3], True)
    except:
        # use GPU 0
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)



if __name__ == "__main__":

    model_name = "4_model"
    save_model_path = "model/" + str(model_name) + ".keras"

    batch_size = 1
    img_size=(1086, 2046)
    epochs = 300
    num_classes = 26

    # Load dataset
    train_loader = DataLoader(batch_size=batch_size, img_size=img_size, split='train', merge_classes=True)
    val_loader = DataLoader(batch_size=batch_size, img_size=img_size, split='val', merge_classes=True)

    # Load model
    model = FastSCNN(num_classes=num_classes, input_shape=(img_size[0], img_size[1], 3))
    # model = create_unet_model()
    # model = DeepLabV3Plus(input_shape=(img_size[0], img_size[1], 3))
    # model = simple_unet(input_shape=(img_size[0], img_size[1], 3))
    # model = create_small_unet_pretrained(input_shape=(img_size[0], img_size[1], 3))
    # model = load_pretrained_hrnet(input_shape=(img_size[0], img_size[1], 3))

    compile_model(model, loss='ohem_loss')
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_loss', mode='min')
    #checkpoint = ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_mean_iou', mode='max')
    #checkpoint = ModelCheckpoint(save_model_path, save_best_only=True,
    #                             monitor='val_sparse_categorical_accuracy', mode='max')
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    reduce_plateau = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)

    # Train model
    model.fit(train_loader, validation_data=val_loader, epochs=epochs,
              callbacks=[checkpoint, reduce_plateau, early_stopping])
