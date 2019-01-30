import numpy as np
from layers.input import InputLayer
from layers.output import OutputLayer
from keras.models import Model

if __name__ == "__main__":
    input_layer = InputLayer()
    inputs = input_layer.create()
    output_layer = OutputLayer(inputs)
    outputs = output_layer.create()
    model = Model(inputs=inputs, outputs=outputs)

    model.summary()
    for layer in model.layers:
        print(layer.output_shape)
    # print(inputs)