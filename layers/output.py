
from keras.layers import Dense
class OutputLayer:

    def __init__(self, input):
        self.input = input
        pass

    def create(self):
        x = Dense(3, activation="softmax")(self.input)
        return x