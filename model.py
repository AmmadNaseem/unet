"""U-Net model for binary or indexed multiclass segmentation.

For the new dataset use input_size=(128, 128, 3), num_classes=3.
Run this file directly to build and summarize that configuration.
"""
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def unet(pretrained_weights=None, input_size=(256,256,1), num_classes=1, base_filters=64, learning_rate=1e-4, dropout_rate=0.5):
    if len(input_size) != 3 or any(not isinstance(n, int) or n < 1 for n in input_size):
        raise ValueError("input_size must contain positive integer height, width, and channels")
    if input_size[0] % 16 or input_size[1] % 16:
        raise ValueError("Input height and width must be divisible by 16 for four pooling levels")
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes must be a positive integer; use 1 for binary segmentation")
    if not isinstance(base_filters, int) or base_filters < 1:
        raise ValueError("base_filters must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be between 0 and 1 (exclusive)")

    inputs = Input(shape=input_size)
    conv1 = Conv2D(base_filters * 1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(base_filters * 1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(base_filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(base_filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(base_filters * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(base_filters * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(base_filters * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(base_filters * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(base_filters * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(base_filters * 16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    up6 = Conv2D(base_filters * 8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(base_filters * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(base_filters * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(base_filters * 4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(base_filters * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(base_filters * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(base_filters * 2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(base_filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(base_filters * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(base_filters * 1, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(base_filters * 1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(base_filters * 1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # Retain the legacy two-feature head for binary checkpoints only.
    # Multiclass output uses the full decoder features instead of a two-feature bottleneck.
    if num_classes == 1:
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(num_classes, 1, activation='sigmoid' if num_classes == 1 else 'softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer = Adam(learning_rate=learning_rate), loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model



if __name__ == "__main__":
    unet(input_size=(128, 128, 3), num_classes=3, base_filters=16).summary()
