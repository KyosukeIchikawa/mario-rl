from typing import Optional

import numpy as np
import tensorflow as tf


class NoisyDense(tf.keras.layers.Layer):
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
        self.activation = tf.keras.activations.get(activation)
        self._sigma_0 = sigma_0
        self._mu_w = None
        self._sigma_w = None
        self._epsilon_w = None
        self._mu_b = None
        self._sigma_b = None
        self._epsilon_b = None

    def build(self, input_shape):
        """Builds the layer."""
        n_input = input_shape[-1]
        init_mu_min = -1/np.sqrt(n_input)
        init_mu_max = 1/np.sqrt(n_input)
        init_sigma = self._sigma_0 / np.sqrt(n_input)
        self._mu_w = self.add_weight(shape=(n_input, self.units),
                                     initializer=tf.keras.initializers.RandomUniform(init_mu_min, init_mu_max),
                                     trainable=self.trainable, name='mu_w')
        self._sigma_w = self.add_weight(shape=(n_input, self.units),
                                        initializer=tf.keras.initializers.Constant(init_sigma),
                                        trainable=self.trainable, name='sigma_w')
        self._mu_b = self.add_weight(shape=(self.units,),
                                     initializer=tf.keras.initializers.RandomUniform(init_mu_min, init_mu_max),
                                     trainable=True, name='mu_b')
        self._sigma_b = self.add_weight(shape=(self.units,),
                                        initializer=tf.keras.initializers.Constant(init_sigma),
                                        trainable=True, name='sigma_b')

        dtype = self._mu_w.dtype
        epsilon_in = self._f(tf.random.normal(shape=tf.shape(self._mu_w.shape[0], 1), dtype=dtype))
        epsilon_out = self._f(tf.random.normal(shape=tf.shape(1, self._mu_w.shape[1]), dtype=dtype))
        self._epsilon_w = tf.matmul(epsilon_in, epsilon_out)
        self._epsilon_b = epsilon_out

        super().build(input_shape)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """Calls the layer.

        :param inputs: Input tensor.
        :param kwargs: Additional arguments.
        :return: Output tensor.
        """
        w = self._mu_w + self._sigma_w * self._epsilon_w
        b = self._mu_b + self._sigma_b * self._epsilon_b
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
            'activation': tf.keras.activations.serialize(self.activation),
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
