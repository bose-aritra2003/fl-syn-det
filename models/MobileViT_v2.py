import tensorflow as tf
import keras
from keras_vision.MobileViT_v2 import build_MobileViT_v2

def MobileViT_v2(input_shape=(64, 64, 3)):
    mobilenetv2 = build_MobileViT_v2(
    width_multiplier=2.0,
    input_shape=input_shape,
    pretrained=False,
    num_classes =1
)
    
    return mobilenetv2


model = MobileViT_v2(input_shape=(64, 64, 3))
model.summary()