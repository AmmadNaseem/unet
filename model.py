"""Configurable TensorFlow-Keras U-Net and serializable CCE + Dice loss."""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


@tf.keras.utils.register_keras_serializable(package="UNet")
class CategoricalDiceLoss(tf.keras.losses.Loss):
    """L = mean categorical CE + dice_weight * (1 - mean soft Dice)."""
    def __init__(self, dice_weight=1.0, smooth=1e-6, name="categorical_dice_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice_weight = dice_weight
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
        intersection = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(0, 1, 2))
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return ce + self.dice_weight * (1 - tf.reduce_mean(dice))

    def get_config(self):
        return {**super().get_config(), "dice_weight": self.dice_weight, "smooth": self.smooth}


def unet(pretrained_weights=None, input_size=(256,256,1), num_classes=1,
         base_filters=64, learning_rate=1e-4, dropout_rate=0.5,
         depth=4, loss_mode="sparse", dice_weight=1.0):
    """depth counts pooling/upsampling levels; 4 gives the standard U-Net."""
    if not isinstance(depth, int) or not 1 <= depth <= 5:
        raise ValueError("depth must be an integer between 1 and 5")
    if len(input_size) != 3 or any(not isinstance(n, int) or n < 1 for n in input_size):
        raise ValueError("input_size must contain three positive integers")
    if input_size[0] % (2**depth) or input_size[1] % (2**depth):
        raise ValueError("Height and width must be divisible by 2**depth")
    if not isinstance(base_filters, int) or base_filters < 1 or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("Filters and class count must be positive integers")
    if learning_rate <= 0 or not 0 <= dropout_rate < 1 or dice_weight < 0:
        raise ValueError("Invalid learning rate, dropout, or Dice weight")
    if loss_mode not in {"sparse", "cce", "cce_dice"}:
        raise ValueError("loss_mode must be sparse, cce, or cce_dice")

    def block(x, filters):
        x = Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        return Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer="he_normal")(x)

    inputs = Input(shape=input_size)
    x = inputs
    skips = []
    for level in range(depth):
        x = block(x, base_filters * 2**level)
        if level == depth - 1:
            x = Dropout(dropout_rate)(x)
        skips.append(x)
        x = MaxPooling2D((2,2))(x)
    x = Dropout(dropout_rate)(block(x, base_filters * 2**depth))
    for level in reversed(range(depth)):
        x = UpSampling2D((2,2))(x)
        x = Conv2D(base_filters * 2**level, 2, activation="relu", padding="same", kernel_initializer="he_normal")(x)
        x = block(concatenate([skips[level], x]), base_filters * 2**level)
    if num_classes == 1:
        x = Conv2D(2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(x)
    outputs = Conv2D(num_classes, 1, activation="sigmoid" if num_classes == 1 else "softmax")(x)
    model = Model(inputs, outputs)
    if num_classes == 1:
        loss = "binary_crossentropy"
    elif loss_mode == "sparse":
        loss = "sparse_categorical_crossentropy"
    elif loss_mode == "cce":
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss = CategoricalDiceLoss(dice_weight=dice_weight)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, metrics=["accuracy"], jit_compile=False)
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
    return model


if __name__ == "__main__":
    unet(input_size=(128,128,3), num_classes=3, base_filters=16, loss_mode="cce").summary()
