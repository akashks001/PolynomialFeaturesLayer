import tensorflow as tf

class PolynomialFeaturesLayer(tf.keras.layers.Layer):
    """
    This creates Polynomial features of given input.

    Example:
    Input = [x1, x2]
    Output = [[x1, x2, x1*x1, x1*x2, x2*x2]]

    This creates all combinations of multiples of input variables.
    This is a desinged as a Tensorflow layer.
    It can be used just before Dense Layer or any other layer.

    This is expected to provide the speed and agility of Tensorflow instead of
    using sklearn polynomial features.

    ALGORITHM
    ---------

    Consider a vector A (which is supposed to be the input to the model) and its transpose At
    A = [[x1, x2]]
    At = [[x1],
         [[x2]]

    At*A = [[x1*x1, x1*x2],
            [x2*x1, x2*x2]]

    Consider lower triangle matrix of At*A

    At*A = [[x1*x1, 0],
            [x2*x1, x2*x2]]

    When we flatten At*A, we get [x1*x1, 0, x2*x1, x2*x2].

    In implementation, the entire operation above is first done using a matrix of only ones.
    After taking matrix multiplication and lower triangle matrix, the indices of non-zero elements
    are extracted.

    These indices are then applied to At*A and then expanded to get flattened [[x1, x2, x1*x1, x1*x2, x2*x2]]
    """
    def __init__(self, power=4):
        super(PolynomialFeaturesLayer, self).__init__()
        self.power = power

    def build(self, input_shape):
        self.indices_to_extract = tf.ones((input_shape[-1], input_shape[-1]))
        self.indices_to_extract = tf.linalg.band_part(self.indices_to_extract, -1,
                                                      0)  # Getting lower triangle part of self.indices to extract
        self.indices_to_extract = tf.reshape(self.indices_to_extract, (1, -1))[0]  # Making matrix into row matrix
        self.indices_to_extract = tf.where(
            self.indices_to_extract == 1)  # Getting indices inside tensor which has value equal to one
        self.indices_to_extract = tf.reshape(self.indices_to_extract,
                                             (1, 1, -1))  # Expanding the indices matrix to suit matrix extraction

    def call(self, inputs):
        if (len(inputs.shape) == 2):
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], 1, tf.shape(inputs)[1]))
        self.X = tf.matmul(inputs, inputs, transpose_a=True)
        self.gather_indices = tf.repeat(self.indices_to_extract, tf.shape(inputs)[0],
                                        axis=0)  # Repeating self.indices_to_extract same number of times as input shape
        self.X = tf.reshape(self.X, (tf.shape(inputs)[0], 1, -1))  # Flattening the multiplied matrices
        self.X = tf.gather(self.X, self.gather_indices,
                           batch_dims=2)  # Select only those entries of x whose indices are in self.gather_indices
        outputs = tf.concat([inputs, self.X], axis=2)  # Now create new inputs matrix which has x1
        outputs = tf.reshape(outputs, (tf.shape(outputs)[0], tf.shape(outputs)[-1]))
        return outputs


"""
Usage Example 

import numpy as np

#The Dataset
x1 = np.linspace(0,10,2000)
x1 = np.reshape(x1,(-1,1))
x2 = np.linspace(10,20,2000)
x2 = np.reshape(x2,(-1,1))
x = np.concatenate((x1,x2), axis=1)
y = 2*x1+3*x2+5*x1*x2+x1**3+x2**2 

#Using in a Model
inputs = tf.keras.layers.Input(shape=(2))
layer = PolynomialFeaturesLayer(power=2)(inputs)
dense = tf.keras.layers.Dense(5)(layer)
outputs = tf.keras.layers.Dense(1)(dense)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss = 'mse')
model.summary()

model.fit(x,y,batch_size=10,epochs=100)
"""
