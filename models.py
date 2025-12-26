"""
Neural network architectures for emergent language learning.

This module implements the core neural network components:
- Encoder for processing observations
- Message generation using Gumbel-Softmax for differentiable discrete messages
- Message interpretation network
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class GumbelSoftmax(layers.Layer):
    """
    Gumbel-Softmax layer for differentiable sampling from categorical distributions.
    
    The Gumbel-Softmax trick allows us to sample from a categorical distribution
    in a differentiable way, which is crucial for training discrete communication.
    """
    
    def __init__(self, temperature=1.0, hard=False, **kwargs):
        """
        Args:
            temperature: Softmax temperature (lower = more discrete)
            hard: If True, use straight-through estimator for one-hot outputs
        """
        super(GumbelSoftmax, self).__init__(**kwargs)
        self.temperature = temperature
        self.hard = hard
    
    def call(self, logits, training=None):
        """
        Apply Gumbel-Softmax to logits.
        
        Args:
            logits: Tensor of shape [..., num_classes]
            training: Whether in training mode
            
        Returns:
            Sampled tensor of same shape as logits
        """
        if training:
            # Sample from Gumbel(0, 1)
            uniform_samples = tf.random.uniform(tf.shape(logits), minval=0, maxval=1)
            gumbel_samples = -tf.math.log(-tf.math.log(uniform_samples + 1e-20) + 1e-20)
            
            # Add Gumbel noise to logits and apply softmax
            y = tf.nn.softmax((logits + gumbel_samples) / self.temperature)
            
            if self.hard:
                # Straight-through estimator: forward pass is one-hot, backward pass is soft
                y_hard = tf.cast(tf.one_hot(tf.argmax(y, axis=-1), tf.shape(logits)[-1]), y.dtype)
                y = tf.stop_gradient(y_hard - y) + y
            
            return y
        else:
            # During inference, just return argmax (greedy decoding)
            return tf.one_hot(tf.argmax(logits, axis=-1), tf.shape(logits)[-1])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'hard': self.hard
        })
        return config


class ObservationEncoder(keras.Model):
    """
    Encodes observations into a fixed-size representation.
    """
    
    def __init__(self, observation_dim, hidden_dim=128, embedding_dim=64, **kwargs):
        """
        Args:
            observation_dim: Dimensionality of input observations
            hidden_dim: Size of hidden layers
            embedding_dim: Size of output embeddings
        """
        super(ObservationEncoder, self).__init__(**kwargs)
        
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.dense3 = layers.Dense(embedding_dim, activation='relu')
        
    def call(self, observations):
        """
        Encode observations.
        
        Args:
            observations: Tensor of shape [batch_size, observation_dim]
            
        Returns:
            Embeddings of shape [batch_size, embedding_dim]
        """
        x = self.dense1(observations)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class MessageGenerator(keras.Model):
    """
    Generates discrete messages using Gumbel-Softmax.
    """
    
    def __init__(self, vocab_size, message_length, hidden_dim=128, 
                 temperature=1.0, hard=True, **kwargs):
        """
        Args:
            vocab_size: Number of discrete symbols in vocabulary
            message_length: Length of messages to generate
            hidden_dim: Size of hidden layers
            temperature: Gumbel-Softmax temperature
            hard: Whether to use straight-through estimator
        """
        super(MessageGenerator, self).__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.message_length = message_length
        
        # Dense layer to transform embeddings
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        
        # Output layers for each position in the message
        self.message_outputs = [
            layers.Dense(vocab_size) for _ in range(message_length)
        ]
        
        # Gumbel-Softmax layers
        self.gumbel_softmax = GumbelSoftmax(temperature=temperature, hard=hard)
    
    def call(self, embeddings, training=None):
        """
        Generate messages from embeddings.
        
        Args:
            embeddings: Tensor of shape [batch_size, embedding_dim]
            training: Whether in training mode
            
        Returns:
            Messages of shape [batch_size, message_length, vocab_size]
        """
        x = self.dense1(embeddings)
        
        # Generate each symbol in the message
        message_symbols = []
        for output_layer in self.message_outputs:
            logits = output_layer(x)
            symbol = self.gumbel_softmax(logits, training=training)
            message_symbols.append(symbol)
        
        # Stack symbols to form complete message
        message = tf.stack(message_symbols, axis=1)
        return message
    
    def set_temperature(self, temperature):
        """Update the Gumbel-Softmax temperature."""
        self.gumbel_softmax.temperature = temperature


class MessageInterpreter(keras.Model):
    """
    Interprets messages to predict observations.
    """
    
    def __init__(self, vocab_size, message_length, hidden_dim=128, 
                 output_dim=64, **kwargs):
        """
        Args:
            vocab_size: Number of discrete symbols in vocabulary
            message_length: Length of messages
            hidden_dim: Size of hidden layers
            output_dim: Dimensionality of output predictions
        """
        super(MessageInterpreter, self).__init__(**kwargs)
        
        # Flatten the message representation
        self.flatten = layers.Flatten()
        
        # Process the message
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.dense3 = layers.Dense(output_dim, activation='relu')
    
    def call(self, messages):
        """
        Interpret messages to produce predictions.
        
        Args:
            messages: Tensor of shape [batch_size, message_length, vocab_size]
            
        Returns:
            Predictions of shape [batch_size, output_dim]
        """
        x = self.flatten(messages)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class CommunicationModel(keras.Model):
    """
    Complete communication model combining Speaker and Listener.
    """
    
    def __init__(self, observation_dim, vocab_size, message_length,
                 hidden_dim=128, embedding_dim=64, temperature=1.0, **kwargs):
        """
        Args:
            observation_dim: Dimensionality of observations
            vocab_size: Size of vocabulary
            message_length: Length of messages
            hidden_dim: Size of hidden layers
            embedding_dim: Size of embeddings
            temperature: Initial Gumbel-Softmax temperature
        """
        super(CommunicationModel, self).__init__(**kwargs)
        
        self.encoder = ObservationEncoder(observation_dim, hidden_dim, embedding_dim)
        self.speaker = MessageGenerator(vocab_size, message_length, hidden_dim, temperature)
        self.listener = MessageInterpreter(vocab_size, message_length, hidden_dim, embedding_dim)
    
    def call(self, observations, training=None):
        """
        Full forward pass: observation -> message -> interpretation.
        
        Args:
            observations: Tensor of shape [batch_size, observation_dim]
            training: Whether in training mode
            
        Returns:
            Tuple of (messages, interpretations)
        """
        embeddings = self.encoder(observations)
        messages = self.speaker(embeddings, training=training)
        interpretations = self.listener(messages)
        return messages, interpretations
