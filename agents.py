"""
Speaker and Listener agent classes for emergent language learning.

This module defines the two agents that communicate:
- Speaker: Observes an object and generates a message
- Listener: Receives a message and predicts the object
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from models import ObservationEncoder, MessageGenerator, MessageInterpreter


class Speaker:
    """
    Speaker agent that observes objects and generates messages.
    """
    
    def __init__(self, observation_dim, vocab_size, message_length,
                 hidden_dim=128, embedding_dim=64, temperature=1.0,
                 learning_rate=0.001):
        """
        Initialize Speaker agent.
        
        Args:
            observation_dim: Dimensionality of observations
            vocab_size: Size of vocabulary
            message_length: Length of messages to generate
            hidden_dim: Size of hidden layers
            embedding_dim: Size of embeddings
            temperature: Initial Gumbel-Softmax temperature
            learning_rate: Learning rate for optimizer
        """
        self.observation_dim = observation_dim
        self.vocab_size = vocab_size
        self.message_length = message_length
        
        # Build Speaker networks
        self.encoder = ObservationEncoder(observation_dim, hidden_dim, embedding_dim)
        self.message_generator = MessageGenerator(
            vocab_size, message_length, hidden_dim, temperature, hard=True
        )
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    def get_message(self, observation, training=True):
        """
        Generate a message from an observation.
        
        Args:
            observation: Tensor of shape [batch_size, observation_dim]
            training: Whether in training mode
            
        Returns:
            Message of shape [batch_size, message_length, vocab_size]
        """
        embedding = self.encoder(observation)
        message = self.message_generator(embedding, training=training)
        return message
    
    def get_trainable_variables(self):
        """Get all trainable variables."""
        return (self.encoder.trainable_variables + 
                self.message_generator.trainable_variables)
    
    def set_temperature(self, temperature):
        """Update the Gumbel-Softmax temperature."""
        self.message_generator.set_temperature(temperature)


class Listener:
    """
    Listener agent that receives messages and predicts objects.
    """
    
    def __init__(self, vocab_size, message_length, observation_dim,
                 hidden_dim=128, embedding_dim=64, learning_rate=0.001):
        """
        Initialize Listener agent.
        
        Args:
            vocab_size: Size of vocabulary
            message_length: Length of messages
            observation_dim: Dimensionality of observations to predict
            hidden_dim: Size of hidden layers
            embedding_dim: Size of output embeddings
            learning_rate: Learning rate for optimizer
        """
        self.vocab_size = vocab_size
        self.message_length = message_length
        self.observation_dim = observation_dim
        
        # Build Listener networks
        self.message_interpreter = MessageInterpreter(
            vocab_size, message_length, hidden_dim, embedding_dim
        )
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    def interpret_message(self, message):
        """
        Interpret a message to predict an observation.
        
        Args:
            message: Tensor of shape [batch_size, message_length, vocab_size]
            
        Returns:
            Prediction of shape [batch_size, embedding_dim]
        """
        prediction = self.message_interpreter(message)
        return prediction
    
    def get_trainable_variables(self):
        """Get all trainable variables."""
        return self.message_interpreter.trainable_variables


class CommunicationAgents:
    """
    Container class for both Speaker and Listener agents.
    """
    
    def __init__(self, observation_dim, vocab_size, message_length,
                 hidden_dim=128, embedding_dim=64, temperature=1.0,
                 learning_rate=0.001):
        """
        Initialize both agents.
        
        Args:
            observation_dim: Dimensionality of observations
            vocab_size: Size of vocabulary
            message_length: Length of messages
            hidden_dim: Size of hidden layers
            embedding_dim: Size of embeddings
            temperature: Initial Gumbel-Softmax temperature
            learning_rate: Learning rate for optimizers
        """
        self.speaker = Speaker(
            observation_dim, vocab_size, message_length,
            hidden_dim, embedding_dim, temperature, learning_rate
        )
        
        self.listener = Listener(
            vocab_size, message_length, observation_dim,
            hidden_dim, embedding_dim, learning_rate
        )
        
        self.vocab_size = vocab_size
        self.message_length = message_length
    
    def communicate(self, observation, training=True):
        """
        Full communication: Speaker generates message, Listener interprets.
        
        Args:
            observation: Observation to communicate about
            training: Whether in training mode
            
        Returns:
            Tuple of (message, listener_prediction)
        """
        message = self.speaker.get_message(observation, training=training)
        prediction = self.listener.interpret_message(message)
        return message, prediction
    
    def set_temperature(self, temperature):
        """Update the Gumbel-Softmax temperature for the Speaker."""
        self.speaker.set_temperature(temperature)
    
    def save_weights(self, filepath):
        """Save agent weights to files."""
        self.speaker.encoder.save_weights(f"{filepath}_speaker_encoder.h5")
        self.speaker.message_generator.save_weights(f"{filepath}_speaker_generator.h5")
        self.listener.message_interpreter.save_weights(f"{filepath}_listener_interpreter.h5")
    
    def load_weights(self, filepath):
        """Load agent weights from files."""
        self.speaker.encoder.load_weights(f"{filepath}_speaker_encoder.h5")
        self.speaker.message_generator.load_weights(f"{filepath}_speaker_generator.h5")
        self.listener.message_interpreter.load_weights(f"{filepath}_listener_interpreter.h5")
