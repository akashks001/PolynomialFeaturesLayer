# PolynomialFeaturesLayer
This creates Polynomial features of given input.

Example:
Input = [x1, x2]
Output = [[x1, x2, x1*x1, x1*x2, x2*x2]]

This creates all combinations of multiples of input variables.
This is a desinged as a Tensorflow layer.
It can be used just before Dense Layer or any other layer.

This is expected to provide the speed and agility of Tensorflow instead of
using sklearn polynomial features.
