import tensorflow as tf
from training.semantic.dataloader import SemanticDataLoader
from training.semantic.model import compile_model,load_pretrained_hrnet
import numpy as np
from tqdm import tqdm



def pixel_accuracy(model, dataset, num_classes, max_samples=None):
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_samples = 0

    progress = tqdm(total=max_samples if max_samples else sum(1 for _ in dataset), desc="Validating", unit="img")

    for image_batch, mask_batch in dataset:
        if max_samples is not None and total_samples >= max_samples:
            break

        batch_size = image_batch.shape[0]
        if max_samples is not None and total_samples + batch_size > max_samples:
            batch_size = max_samples - total_samples
            image_batch = image_batch[:batch_size]
            mask_batch = mask_batch[:batch_size]

        pred_logits = model.predict(image_batch, verbose=0)
        pred_labels = np.argmax(pred_logits, axis=-1)
        true_labels = mask_batch.squeeze()

        # Compute confusion matrix
        pred_flat = pred_labels.reshape(-1)
        true_flat = true_labels.reshape(-1)
        cm = tf.math.confusion_matrix(true_flat, pred_flat, num_classes=num_classes, dtype=tf.dtypes.int64).numpy()
        total_cm += cm

        total_samples += batch_size
        progress.update(batch_size)

    progress.close()

    # Compute accuracy metrics
    diag = np.diag(total_cm)

    return {"pixel_accuracy": float(diag.sum() / total_cm.sum())}


if __name__ == "__main__":
    batch_size = 1
    img_size = (1086, 2046)
    num_classes = 13  # 13, 19,25 -> set merge_classes in dataloader

    model = load_pretrained_hrnet(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)

    # load weights
    if num_classes == 25:
        merge = False
        model.load_weights("model/hr_model_25_classes.keras")
    elif num_classes == 19:
        merge = True
        model.load_weights("model/hr_model_19_classes.keras")
    elif num_classes == 13:
        merge = True
        model.load_weights("model/hr_model_13_classes.keras")


    compile_model(model)
    model.summary()

    test_loader = SemanticDataLoader(batch_size=1, img_size=img_size, split='test', num_classes=num_classes, shuffle=False)


    # print pixel accuracy
    print(pixel_accuracy(model, test_loader, num_classes, max_samples=len(test_loader)))

    # print loss, accuracy and miou
    results = model.evaluate(test_loader)
    print("Test Loss:", results[0])
    print("Test Accuracy:", results[1])
    print("Test mIoU:", results[2])

    test_loader.visualize_results(model, img_size=img_size)


