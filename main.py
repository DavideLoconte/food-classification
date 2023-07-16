import shutil
import os

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
NUM_CLASSES = 101

checkpoint_path = "model"


def get_classes() -> list:
    return [x for x in os.listdir('train') if os.path.isdir(f"train/{x}")]


def get_model(classes):
    if os.path.exists(checkpoint_path):
        return tf.keras.models.load_model(checkpoint_path)

    img_augmentation = Sequential(
        [
            preprocessing.RandomFlip("horizontal"),
            preprocessing.RandomRotation(0.2),
            preprocessing.RandomHeight(0.2),
            preprocessing.RandomWidth(0.2),
            preprocessing.RandomZoom(0.2)
        ],
        name="img_augmentation",
    )

    base_model = tf.keras.applications.EfficientNetV2S(include_top=False)
    base_model.trainable = False
    inputs = layers.Input(
        shape=INPUT_SHAPE,
        name="input_layer")  # shape of input image

    x = img_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    outputs = layers.Dense(len(classes), activation="softmax", name="output_layer")(
        x)  # same number of outputs as classes

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy",
        metrics=["accuracy"])

    return model


model = get_model(get_classes())

train_batches = tf.keras.preprocessing.image_dataset_from_directory(
    "train",
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    image_size=(INPUT_SHAPE[0],
                INPUT_SHAPE[1])
)

test_batches = tf.keras.preprocessing.image_dataset_from_directory(
    "test",
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
    image_size=(INPUT_SHAPE[0],
                INPUT_SHAPE[1]))

validation_batches = tf.keras.preprocessing.image_dataset_from_directory(
    "validation",
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
    image_size=(INPUT_SHAPE[0],
                INPUT_SHAPE[1]))


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1,
                                                 save_best_only=True)

_ = model.fit(train_batches, epochs=200, verbose=1, callbacks=[
              cp_callback], validation_data=validation_batches)
