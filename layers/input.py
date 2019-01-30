from keras.layers import Input

class InputLayer:

    def __init__(self, size=10, dimension=10):
        self.size = size
        self.dimension = dimension

    def create(self):
        input = Input(shape=(self.size, self.dimension,))
        return input
