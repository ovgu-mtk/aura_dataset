import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
def ohem_loss(y_true, y_pred, k=0.7):
    # Convert ground truth to one-hot encoding
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=26)

    # Calculate per-pixel cross entropy loss
    pixel_losses = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Flatten the losses
    pixel_losses_flat = tf.reshape(pixel_losses, [-1])

    # Sort losses and get k% hardest examples
    k_pixels = tf.cast(tf.round(tf.cast(tf.size(pixel_losses_flat), tf.float32) * k), tf.int32)

    # Get the hardest k% pixels
    hardest_pixels = tf.sort(pixel_losses_flat, direction='DESCENDING')[:k_pixels]

    # Return mean of hardest pixels' losses
    return tf.reduce_mean(hardest_pixels)

@tf.keras.saving.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1.0):
    # Convert the target from (None, 512, 1024, 1) with integer values to one-hot encoded (None, 512, 1024, 26)
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.squeeze(y_true, axis=-1)  # Remove the last dimension to get shape (None, 512, 1024)
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)  # Now shape (None, 512, 1024, 26)

    # Flatten the tensors
    y_true_f = tf.reshape(y_true_one_hot, [-1, num_classes])
    y_pred_f = tf.reshape(y_pred, [-1, num_classes])

    # Calculate intersection and union for each class
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)

    # Calculate Dice coefficient for each class
    dice_coef = (2. * intersection + smooth) / (union + smooth)

    # Calculate mean Dice loss
    dice_loss = 1 - tf.reduce_mean(dice_coef)

    return dice_loss


def combined_loss(y_true, y_pred, alpha=0.5):
    return alpha * dice_loss(y_true, y_pred) + (1 - alpha) * ohem_loss(y_true, y_pred)