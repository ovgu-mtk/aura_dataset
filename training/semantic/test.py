import tensorflow as tf
from dataloader import DataLoader
from tensorflow_model import compile_model, FastSCNN, create_unet_model, create_small_unet_pretrained, DeepLabV3Plus, \
        load_pretrained_hrnet, simple_unet
from aura_dataset.training.semantic.utils.metrics import MeanIoU
from aura_dataset.training.semantic.utils.losses import combined_loss, dice_loss, ohem_loss


# fasten inference speed
@tf.function
def infer(model, X_input):
    return model(X_input, training=False)

# Example usage
if __name__ == "__main__":
    images_path = "../../dataset/semantic/test/images"
    annotations_path = "../../dataset/semantic/test/annotations"
    batch_size = 1
    #img_size=(512, 1024)
    img_size = (1086, 2046)
    num_classes = 26

    model = FastSCNN(num_classes=num_classes, input_shape=(img_size[0], img_size[1], 3))
    # model = create_unet_model()
    # model = simple_unet(input_shape=(512, 1024, 3))
    # model = DeepLabV3Plus(input_shape=(512, 1024, 3))
    # model = create_small_unet_pretrained(input_shape=(512, 1024, 3))
    # model = load_pretrained_hrnet(input_shape=(512, 1024, 3))
    # model = simple_unet(input_shape=(512, 1024, 3))
    # model = load_pretrained_hrnet(input_shape=(512, 1024, 3))

    # load weights
    model.load_weights("model/best_model.keras")
    compile_model(model)
    model.summary()


    test_loader = DataLoader(images_path=images_path,
                             annotations_path=annotations_path,
                             batch_size=1,
                             img_size=img_size,
                             split='test',
                             merge_classes=True,
                             shuffle=False)


    results = model.evaluate(test_loader)
    print("Test Loss:", results[0])
    print("Test Accuracy:", results[1])
    print("Test mIoU:", results[2])


    test_loader.visualize_results(model, images_path, annotations_path, img_size=img_size)


