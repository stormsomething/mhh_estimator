from keras.layers import Layer
from keras import backend as K

class SumLayer(Layer):
    """
    Simple sum layer necessary for deepsets. 

    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on their
    own

    For now, its only compatible with python3

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.supports_masking = True
        # self.name = "name_of_layer"

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if mask is not None:
            x = x * K.cast(mask, K.dtype(x))[:,:,None]
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask):
        return None
