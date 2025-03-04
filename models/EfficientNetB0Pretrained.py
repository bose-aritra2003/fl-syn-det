import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D

def EfficientNetB0Pretrained(input_shape=(64, 64, 3)):
    # Load pre-trained EfficientNetB0 without the top layer
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    
    # Reduce complexity since input is smaller
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model
