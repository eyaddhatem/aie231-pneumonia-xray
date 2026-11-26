"""Model builders shared by notebooks, training and inference. Scaling lives INSIDE the models."""
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 224


def build_cnn(dropout: float = 0.3, batchnorm: bool = False):
    """Small CNN from scratch: 3 x (Conv3x3 + MaxPool) -> GAP -> Dense(64) -> Dropout -> sigmoid (27,809 params)."""
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1 / 255)(inp)
    for f in (16, 32, 64):
        x = layers.Conv2D(f, 3, padding="same", activation=None if batchnorm else "relu")(x)
        if batchnorm:
            x = layers.Activation("relu")(layers.BatchNormalization()(x))
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="cnn_small")


def build_vgg(dropout: float = 0.5, pretrained: bool = True):
    """VGG16 + GAP + Dropout + Dense(1, sigmoid). Base frozen. Returns (model, base)."""
    base = keras.applications.VGG16(include_top=False, weights="imagenet" if pretrained else None,
                                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Lambda(keras.applications.vgg16.preprocess_input)(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="vgg16"), base


def unfreeze_block5(model, base):
    """Fine-tune only VGG16 block5 (call before re-compiling with a small learning rate)."""
    for layer in base.layers:
        layer.trainable = layer.name.startswith("block5")
    base.trainable = True
