import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetV2B0

def EfficientNetV2B0_Custom():
    base_model = EfficientNetV2B0(input_shape=(64, 64, 3), include_top=False, weights=None)  # No pretraining
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification

    model = keras.models.Model(inputs=base_model.input, outputs=output)

    return model