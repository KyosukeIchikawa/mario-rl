from typing import Optional

import keras
import numpy as np
import tensorflow as tf


class NoisyDense(keras.layers.Layer):
    """Noisy dense layer.
    See https://arxiv.org/abs/1706.10295 for details.
    """
    def __init__(self, units: int, activation: Optional[str] = None, sigma_0=0.5, **kwargs):
        """
        :param units: Number of neurons.
        :param activation: Activation function.
        :param kwargs: Additional arguments to pass to the keras.layers.Layer superclass.
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self._sigma_0 = sigma_0
        self._mu_w = None
        self._sigma_w = None
        self._mu_b = None
        self._sigma_b = None

    def build(self, input_shape):
        """Builds the layer."""
        n_input = input_shape[-1]
        init_mu_min = -1/np.sqrt(n_input)
        init_mu_max = 1/np.sqrt(n_input)
        init_sigma = self._sigma_0 / np.sqrt(n_input)
        self._mu_w = self.add_weight(shape=(n_input, self.units),
                                     initializer=keras.initializers.RandomUniform(init_mu_min, init_mu_max),
                                     trainable=self.trainable, name='mu_w')
        self._sigma_w = self.add_weight(shape=(n_input, self.units),
                                        initializer=keras.initializers.Constant(init_sigma),
                                        trainable=self.trainable, name='sigma_w')
        self._mu_b = self.add_weight(shape=(self.units,),
                                     initializer=keras.initializers.RandomUniform(init_mu_min, init_mu_max),
                                     trainable=True, name='mu_b')
        self._sigma_b = self.add_weight(shape=(self.units,),
                                        initializer=keras.initializers.Constant(init_sigma),
                                        trainable=True, name='sigma_b')

        super().build(input_shape)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """Calls the layer.

        :param inputs: Input tensor.
        :param kwargs: Additional arguments.
        :return: Output tensor.
        """
        # Resample factorized Gaussian noise on every forward pass (NoisyNet).
        dtype = self._mu_w.dtype
        n_input = self._mu_w.shape[0]
        epsilon_in = self._f(tf.random.normal(shape=(n_input, 1), dtype=dtype))  # [n_input, 1]
        epsilon_out = self._f(tf.random.normal(shape=(1, self.units), dtype=dtype))  # [1, units]
        epsilon_w = epsilon_in * epsilon_out  # [n_input, units]
        epsilon_b = tf.squeeze(epsilon_out, axis=0)  # [units]
        w = self._mu_w + self._sigma_w * epsilon_w
        b = self._mu_b + self._sigma_b * epsilon_b
        output = tf.matmul(inputs, w) + b
        if self.activation is not None:
            output = self.activation(output)
        return output

    @staticmethod
    def _f(x: tf.Tensor) -> tf.Tensor:
        """Applies the f function to the given tensor.

        :param x: Input tensor.
        :return: Output tensor.
        """
        return tf.multiply(tf.sign(x), tf.sqrt(tf.abs(x)))

    def get_config(self):
        """Returns the configuration of the layer."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'sigma_0': self._sigma_0
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its configuration."""
        return cls(**config)

    @property
    def sigma_0(self):
        """Returns the sigma_0 parameter."""
        return self._sigma_0

    @property
    def mu_w(self):
        """Returns the mu_w parameter."""
        return self._mu_w

    @property
    def sigma_w(self):
        """Returns the sigma_w parameter."""
        return self._sigma_w

    @property
    def mu_b(self):
        """Returns the mu_b parameter."""
        return self._mu_b

    @property
    def sigma_b(self):
        """Returns the sigma_b parameter."""
        return self._sigma_b
