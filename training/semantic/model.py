import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from training.semantic.utils.metrics import MeanIoU
from training.semantic.utils.losses import combined_loss, dice_loss, ohem_loss



# Load a pretrained HRNet model and modify it for input/output shape
def load_pretrained_hrnet(input_shape=(1086, 2046, 3), num_classes=25):
    hrnet_url = "https://tfhub.dev/google/HRNet/ade20k-hrnetv2-w48/1"  # Replace with correct HRNet URL
    base_model = hub.KerasLayer(hrnet_url, trainable=True)

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)

    # Add a Conv2D layer to produce the desired output shape
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax', name="output_layer")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# DeeplabV3Plus

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False):
    x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same",
                      use_bias=use_bias, kernel_initializer=tf.keras.initializers.HeNormal())(block_input)
    x = layers.BatchNormalization()(x)
    return tf.keras.activations.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(input_shape=(1086, 2046, 3), num_classes=25):
    """ DeepLabV3+ with MobileNetV2 backbone """
    model_input = layers.Input(shape=input_shape)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = resnet50.get_layer("conv2_block3_2_relu").output
    input_a = convolution_block(input_a, num_filters=48, kernel_size=1)

    # Calculate the upsampling size to match input_b dimensions
    input_a_shape = input_a.shape

    # Directly upsample to match input_b dimensions
    input_b = layers.UpSampling2D(
        size=(input_a_shape[1] // x.shape[1], input_a_shape[2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)

    # Upsample to original input size
    x = tf.image.resize(x, size=(input_shape[0], input_shape[1]), method='bilinear')
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), activation='softmax', padding="same")(x)
    return tf.keras.Model(inputs=model_input, outputs=model_output)



# PSPNet
def conv_block(x, filters, kernel_size=3, strides=1, dilation_rate=1, use_bias=False, name=None):
    """Convolution block with BatchNorm and ReLU"""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        name=f"{name}_conv" if name else None
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)
    return x


def pyramid_pooling_module(x, pool_sizes=[1, 2, 3, 6], filters=512, input_shape=None, name="ppm"):
    """Pyramid Pooling Module"""
    # Get feature maps at different scales
    pyramid_features = []

    # Get the spatial dimensions of the input feature map
    feature_shape = tf.shape(x)
    h, w = feature_shape[1], feature_shape[2]

    for i, pool_size in enumerate(pool_sizes):
        if pool_size == 1:
            # Global average pooling
            pooled = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap_{i}")(x)
        else:
            # Adaptive average pooling using different pool sizes
            pooled = layers.Lambda(
                lambda inputs: tf.nn.avg_pool2d(
                    inputs,
                    ksize=[1, pool_size, pool_size, 1],
                    strides=[1, pool_size, pool_size, 1],
                    padding='SAME'
                ),
                name=f"{name}_pool_{i}"
            )(x)

        # 1x1 conv to reduce channels
        pooled = layers.Conv2D(
            filters // len(pool_sizes),
            1,
            use_bias=False,
            name=f"{name}_conv_{i}"
        )(pooled)
        pooled = layers.BatchNormalization(name=f"{name}_bn_{i}")(pooled)
        pooled = layers.ReLU(name=f"{name}_relu_{i}")(pooled)

        # Resize to exactly match the original feature map dimensions
        pooled = layers.Lambda(
            lambda inputs: tf.image.resize(
                inputs[0],
                [inputs[1], inputs[2]],
                method='bilinear'
            ),
            name=f"{name}_resize_{i}"
        )([pooled, h, w])

        pyramid_features.append(pooled)

    # Concatenate all pyramid features with original feature map
    x = layers.Concatenate(name=f"{name}_concat")([x] + pyramid_features)

    # Final conv to reduce channels
    x = conv_block(x, filters, kernel_size=3, name=f"{name}_final")

    return x


def resnet_backbone(input_tensor, output_stride=8):
    """ResNet-50 backbone with dilated convolutions"""

    # Initial convolution
    x = conv_block(input_tensor, 64, kernel_size=7, strides=2, name="conv1")
    x = layers.MaxPooling2D(3, strides=2, padding='same', name="maxpool")(x)

    # ResNet blocks
    # Block 1
    for i in range(3):
        if i == 0:
            x = residual_block(x, [64, 64, 256], strides=1, name=f"res2_{i}")
        else:
            x = residual_block(x, [64, 64, 256], name=f"res2_{i}")

    # Block 2
    for i in range(4):
        if i == 0:
            x = residual_block(x, [128, 128, 512], strides=2, name=f"res3_{i}")
        else:
            x = residual_block(x, [128, 128, 512], name=f"res3_{i}")

    # Block 3 (with dilation for output_stride=8)
    for i in range(6):
        if i == 0:
            stride = 1 if output_stride == 8 else 2
            dilation = 2 if output_stride == 8 else 1
            x = residual_block(x, [256, 256, 1024], strides=stride,
                               dilation_rate=dilation, name=f"res4_{i}")
        else:
            dilation = 2 if output_stride == 8 else 1
            x = residual_block(x, [256, 256, 1024], dilation_rate=dilation, name=f"res4_{i}")

    # Block 4 (with dilation)
    for i in range(3):
        if i == 0:
            dilation = 4 if output_stride == 8 else 2
            x = residual_block(x, [512, 512, 2048], strides=1,
                               dilation_rate=dilation, name=f"res5_{i}")
        else:
            dilation = 4 if output_stride == 8 else 2
            x = residual_block(x, [512, 512, 2048], dilation_rate=dilation, name=f"res5_{i}")

    return x


def residual_block(x, filters_list, strides=1, dilation_rate=1, name="res_block"):
    """ResNet residual block"""
    shortcut = x

    # First conv
    x = layers.Conv2D(
        filters_list[0], 1, strides=strides, use_bias=False,
        name=f"{name}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    # Second conv
    x = layers.Conv2D(
        filters_list[1], 3, padding='same', dilation_rate=dilation_rate,
        use_bias=False, name=f"{name}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)

    # Third conv
    x = layers.Conv2D(
        filters_list[2], 1, use_bias=False, name=f"{name}_conv3"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn3")(x)

    # Shortcut connection
    if strides != 1 or shortcut.shape[-1] != filters_list[2]:
        shortcut = layers.Conv2D(
            filters_list[2], 1, strides=strides, use_bias=False,
            name=f"{name}_shortcut_conv"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_relu")(x)

    return x


def create_pspnet(input_shape=(1086, 2046, 3), num_classes=25, output_stride=8):
    """
    Create PSPNet model

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        output_stride: Output stride of the backbone network

    Returns:
        PSPNet model
    """

    # Input
    inputs = layers.Input(shape=input_shape, name="input")

    # Backbone network (ResNet-50 with dilated convolutions)
    backbone_features = resnet_backbone(inputs, output_stride=output_stride)

    # Pyramid Pooling Module
    psp_features = pyramid_pooling_module(
        backbone_features,
        pool_sizes=[1, 2, 3, 6],
        filters=512,
        name="ppm"
    )

    # Final classification layer
    x = layers.Dropout(0.1, name="dropout")(psp_features)
    x = layers.Conv2D(
        num_classes,
        1,
        use_bias=True,
        name="classifier"
    )(x)

    # Upsample to original input size
    x = tf.image.resize(x, size=(input_shape[0], input_shape[1]), method='bilinear')
    outputs = layers.Conv2D(num_classes, kernel_size=(1, 1), activation='softmax', padding="same")(x)

    # Create model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="PSPNet")

    return model


def compile_model(model, num_classes=25, optimizer='adam', loss='combined_loss'):

    if loss == 'categorical_crossentropy':
        loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) # last layer softmax
        return model.compile(optimizer=optimizer,
                             loss= loss_,
                             metrics=[SparseCategoricalAccuracy(), MeanIoU(num_classes=num_classes).mean_iou])
    elif loss == 'combined_loss':
        return model.compile(optimizer=optimizer,
                             loss= combined_loss,
                             metrics=[SparseCategoricalAccuracy(), MeanIoU(num_classes=num_classes).mean_iou])
    elif loss == 'dice_loss':
        return model.compile(optimizer=optimizer,
                             loss= dice_loss,
                             metrics=[SparseCategoricalAccuracy(), MeanIoU(num_classes=num_classes).mean_iou])

    elif loss == 'ohem_loss':
        return model.compile(optimizer=optimizer,
                             loss= ohem_loss,
                             metrics=[SparseCategoricalAccuracy(), MeanIoU(num_classes=num_classes).mean_iou])