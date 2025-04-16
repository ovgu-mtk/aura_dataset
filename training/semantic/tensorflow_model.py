import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
import tensorflow_hub as hub
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from utils.metrics import MeanIoU
from utils.losses import combined_loss, dice_loss, ohem_loss

def ConvBNReLU(x, out_channels, kernel_size=3, stride=1, padding="same"):
    x = layers.Conv2D(out_channels, kernel_size, strides=stride, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def DSConv(x, out_channels, stride=1):
    in_channels = x.shape[-1]
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def DWConv(x, out_channels, stride=1):
    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def LinearBottleneck(x, in_channels, out_channels, t=6, stride=1):
    use_shortcut = (stride == 1 and in_channels == out_channels)
    x_res = x
    x = ConvBNReLU(x, in_channels * t, kernel_size=1, padding='same')
    x = DWConv(x, in_channels * t, stride=stride)
    x = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if use_shortcut:
        x = layers.Add()([x, x_res])
    return x

def make_bottleneck_layers(x, in_channels, out_channels, blocks, t, stride):
    x = LinearBottleneck(x, in_channels, out_channels, t, stride)
    for _ in range(1, blocks):
        x = LinearBottleneck(x, out_channels, out_channels, t, 1)
    return x

def PyramidPooling(x, out_channels):
    input_size = tf.shape(x)[1:3]
    in_channels = x.shape[-1]
    inter_channels = in_channels // 4

    pool_sizes = [1, 2, 3, 6]
    pooled_features = []
    for size in pool_sizes:
        pooled = layers.AveragePooling2D(pool_size=(size, size), strides=(size, size), padding='same')(x)
        pooled = ConvBNReLU(pooled, inter_channels, kernel_size=1, padding='same')
        pooled = tf.image.resize(pooled, input_size)
        pooled_features.append(pooled)

    x = layers.Concatenate()([x] + pooled_features)
    x = ConvBNReLU(x, out_channels, kernel_size=1, padding='same')
    return x

def LearningToDownsample(x):
    x = ConvBNReLU(x, 32, kernel_size=3, stride=2)
    x = DSConv(x, 48, stride=2)
    x = DSConv(x, 64, stride=2)
    return x

def GlobalFeatureExtractor(x):
    x = make_bottleneck_layers(x, 64, 64, 3, t=6, stride=2)
    x = make_bottleneck_layers(x, 64, 96, 3, t=6, stride=2)
    x = make_bottleneck_layers(x, 96, 128, 3, t=6, stride=1)
    x = PyramidPooling(x, 128)
    return x

def FeatureFusion(higher_res_feature, lower_res_feature):
    h, w = tf.shape(higher_res_feature)[1], tf.shape(higher_res_feature)[2]
    lower_res_feature = tf.image.resize(lower_res_feature, [h, w])
    lower_res_feature = DWConv(lower_res_feature, 128, stride=1)
    lower_res_feature = layers.Conv2D(128, 1)(lower_res_feature)
    lower_res_feature = layers.BatchNormalization()(lower_res_feature)

    higher_res_feature = layers.Conv2D(128, 1)(higher_res_feature)
    higher_res_feature = layers.BatchNormalization()(higher_res_feature)

    x = layers.Add()([higher_res_feature, lower_res_feature])
    x = layers.ReLU()(x)
    return x


def Classifier(x, num_classes):
    # Statt num_classes Kanäle zu produzieren, geben wir nur einen Kanal aus.
    x = DSConv(x, 128, stride=1)
    x = DSConv(x, 128, stride=1)
    x = layers.Dropout(0.1)(x)

    # Ausgabe: 1 Kanal, der als Graustufenbild interpretiert wird
    x = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)  # Ein Kanal für Graustufenbild
    return x


def FastSCNN(input_shape=(512, 1024, 3), num_classes=26):
    inputs = layers.Input(shape=input_shape)
    size = tf.shape(inputs)[1:3]

    higher_res_features = LearningToDownsample(inputs)
    x = GlobalFeatureExtractor(higher_res_features)
    x = FeatureFusion(higher_res_features, x)
    x = Classifier(x, num_classes)

    # Skalierung auf ursprüngliche Größe
    x = tf.image.resize(x, size)

    # Behalte den Ausgabewert als Float zwischen 0 und 1 (keine Typumwandlung zu int32)
    model = Model(inputs, x, name="FastSCNN")

    return model


def create_segmentation_model(input_shape=(512, 1024, 3), num_classes=26):
    """
    Create a semantic segmentation model based on the U-Net architecture
    with a pre-trained ResNet50 as the encoder.

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of segmentation classes

    Returns:
        A Keras model for semantic segmentation
    """
    # Define the input layer
    inputs = Input(shape=input_shape)

    # Load pre-trained ResNet50 as encoder (without the top classification layer)
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Freeze the encoder layers to prevent updating pre-trained weights initially
    for layer in base_model.layers:
        layer.trainable = False

    # Extract feature maps from different levels of the encoder
    # We'll use these for skip connections in the U-Net architecture
    s1 = base_model.get_layer("conv1_relu").output  # 256x512
    s2 = base_model.get_layer("conv2_block3_out").output  # 128x256
    s3 = base_model.get_layer("conv3_block4_out").output  # 64x128
    s4 = base_model.get_layer("conv4_block6_out").output  # 32x64

    # Bridge (bottleneck)
    b1 = base_model.get_layer("conv5_block3_out").output  # 16x32

    # Decoder path
    # Upsampling and skip connections
    d1 = decoder_block(b1, s4, 512)  # 32x64
    d2 = decoder_block(d1, s3, 256)  # 64x128
    d3 = decoder_block(d2, s2, 128)  # 128x256
    d4 = decoder_block(d3, s1, 64)  # 256x512

    # Final upsampling to original image size
    outputs = layers.Conv2DTranspose(num_classes, 4, strides=2, activation='softmax', padding='same')(d4)  # 512x1024

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model


def decoder_block(inputs, skip_connection, num_filters):
    """
    Decoder block for the U-Net architecture.

    Args:
        inputs: Input tensor from previous layer
        skip_connection: Skip connection from encoder
        num_filters: Number of filters for conv layers

    Returns:
        Decoder block output
    """
    # Upsampling
    x = layers.Conv2DTranspose(num_filters, 3, strides=2, padding='same')(inputs)

    # Skip connection
    x = Concatenate()([x, skip_connection])

    # First conv block
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second conv block
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Dropout for regularization
    x = layers.Dropout(0.2)(x)

    return x


def create_unet_model(input_shape=(512, 1024, 3), num_classes=26):
    """Create a U-Net model with a MobileNetV2 encoder."""

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Contraction path (encoder)
    # Start with a custom entry block to handle the larger input size
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)  # 256x512

    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)  # 128x256

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)  # 64x128

    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)  # 32x64

    # Middle
    conv5 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv5)

    # Expansion path (decoder)
    up6 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv5)  # 64x128
    concat6 = layers.Concatenate()([up6, conv4])
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat6)
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv6)  # 128x256
    concat7 = layers.Concatenate()([up7, conv3])
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat7)
    conv7 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv7)  # 256x512
    concat8 = layers.Concatenate()([up8, conv2])
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat8)
    conv8 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(32, 2, strides=(2, 2), padding='same')(conv8)  # 512x1024
    concat9 = layers.Concatenate()([up9, conv1])
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(concat9)
    conv9 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_small_unet_pretrained(input_shape=(1086, 2046, 3), num_classes=26):
    """Create a U-Net with pre-trained MobileNetV2 encoder - adjusted for 1086x2046 input size"""

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Layer outputs for skip connections
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project'
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Feature extraction model
    down_stack = Model(inputs=base_model.input, outputs=base_model_outputs)

    # Freeze the base model weights, you can modify as needed
    down_stack.trainable = True

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Apply the feature extraction model
    skips = down_stack(inputs)
    x = skips[-1]

    # Decoder/up-sampling path with skip connections
    for i in reversed(range(len(layer_names) - 1)):
        filters = 256 if i >= 3 else 128 if i >= 1 else 64

        # Upsample
        x = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=3,
            strides=2,
            padding='same'
        )(x)

        if skips[i].shape[1] != x.shape[1] or skips[i].shape[2] != x.shape[2]:
            resized_skips = layers.Resizing(height=x.shape[1], width=x.shape[2])(skips[i])
        else:
            resized_skips = skips[i]

        concat = layers.Concatenate()([x, resized_skips])

        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(concat)
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        )(x)

    # Final upsampling with Cropping
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    # Crop to desired output size if necessary
    x = layers.Cropping2D(((1, 1), (1, 1)))(x)  # Assuming cropping 1 pixel each side

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)



def simple_unet(input_shape=(512, 1024, 3), num_classes=26):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up7 = layers.UpSampling2D(size=(2, 2))(conv4)
    up7 = layers.concatenate([up7, conv3], axis=-1)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.UpSampling2D(size=(2, 2))(conv7)
    up8 = layers.concatenate([up8, conv2], axis=-1)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.UpSampling2D(size=(2, 2))(conv8)
    up9 = layers.concatenate([up9, conv1], axis=-1)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv9)

    return tf.keras.Model(inputs=[inputs], outputs=[outputs])


def DeepLabV3Plus(input_shape=(512, 1024, 3), num_classes=26):
    """ DeepLabV3+ with MobileNetV2 backbone """

    inputs = layers.Input(shape=input_shape)

    # Base model - pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_tensor=inputs,  # <--- this connects your input layer!
        include_top=False,
        weights='imagenet'
    )

    # Get intermediate features for skip connections
    skip_connection = base_model.get_layer('block_3_expand_relu').output
    encoder_output = base_model.output

    # ASPP
    x = layers.Conv2D(256, 1, padding='same', use_bias=False)(encoder_output)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Upsample to match skip connection size
    skip_shape = tf.keras.backend.int_shape(skip_connection)
    encoder_shape = tf.keras.backend.int_shape(encoder_output)

    height_factor = skip_shape[1] // encoder_shape[1]
    width_factor = skip_shape[2] // encoder_shape[2]
    x = layers.UpSampling2D(size=(height_factor, width_factor), interpolation='bilinear')(x)

    # Skip connection processing
    skip = layers.Conv2D(48, 1, padding='same', use_bias=False)(skip_connection)
    skip = layers.BatchNormalization()(skip)
    skip = layers.Activation('relu')(skip)

    # Ensure shapes match
    x_shape = tf.keras.backend.int_shape(x)
    if x_shape[1] != skip_shape[1] or x_shape[2] != skip_shape[2]:
        x = layers.Resizing(skip_shape[1], skip_shape[2])(x)

    # Decoder
    x = layers.Concatenate()([x, skip])
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)

    # Upsample to original input size
    x_shape = tf.keras.backend.int_shape(x)
    height_factor = input_shape[0] // x_shape[1]
    width_factor = input_shape[1] // x_shape[2]
    x = layers.UpSampling2D(size=(height_factor, width_factor), interpolation='bilinear')(x)

    # Final output layer — multi-class softmax
    #outputs = layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# Load a pretrained HRNet model and modify it for input/output shape
def load_pretrained_hrnet(input_shape=(512, 1024, 3)):
    hrnet_url = "https://tfhub.dev/google/HRNet/ade20k-hrnetv2-w48/1"  # Replace with correct HRNet URL
    base_model = hub.KerasLayer(hrnet_url, trainable=True)

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)

    # Add a Conv2D layer to produce the desired output shape
    outputs = tf.keras.layers.Conv2D(26, (1, 1), activation='softmax', name="output_layer")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def compile_model(model, num_classes=26, optimizer='adam', loss='combined_loss'):

    if loss == 'categorical_crossentropy':
        return model.compile(optimizer=optimizer,
                             loss= 'sparse_categorical_crossentropy',
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