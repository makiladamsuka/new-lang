"""
Referential game environment for emergent language learning.

The environment generates random observations and provides a game where:
1. Speaker sees a target observation
2. Speaker generates a message
3. Listener receives the message and a set of candidates
4. Listener must identify which candidate matches the target
"""

import numpy as np
import tensorflow as tf

# Temperature scaling factor for converting similarities to logits
SIMILARITY_TEMPERATURE = 10.0


class ReferentialGameEnvironment:
    """
    Environment for the referential communication game.
    """
    
    def __init__(self, observation_dim, num_candidates=4, seed=None):
        """
        Initialize the environment.
        
        Args:
            observation_dim: Dimensionality of observations
            num_candidates: Number of candidate objects for Listener to choose from
            seed: Random seed for reproducibility
        """
        self.observation_dim = observation_dim
        self.num_candidates = num_candidates
        self.rng = np.random.RandomState(seed)
    
    def generate_observation(self, batch_size=1):
        """
        Generate random observations.
        
        Args:
            batch_size: Number of observations to generate
            
        Returns:
            Numpy array of shape [batch_size, observation_dim]
        """
        # Generate random vectors (could be colors, shapes, etc.)
        observations = self.rng.randn(batch_size, self.observation_dim).astype(np.float32)
        # Normalize to unit vectors for stability
        norms = np.linalg.norm(observations, axis=1, keepdims=True)
        observations = observations / (norms + 1e-8)
        return observations
    
    def create_game(self, batch_size=1):
        """
        Create a referential game instance.
        
        Args:
            batch_size: Number of parallel games
            
        Returns:
            Dictionary containing:
                - 'target': The target observation Speaker sees
                - 'candidates': All candidate observations (including target)
                - 'target_idx': Index of target in candidates
        """
        # Generate target observation
        target = self.generate_observation(batch_size)
        
        # Generate distractor observations
        distractors = self.generate_observation(batch_size * (self.num_candidates - 1))
        distractors = distractors.reshape(batch_size, self.num_candidates - 1, self.observation_dim)
        
        # Combine target and distractors into candidates
        # Randomly place target among candidates
        target_idx = self.rng.randint(0, self.num_candidates, size=batch_size)
        
        candidates = np.zeros((batch_size, self.num_candidates, self.observation_dim), dtype=np.float32)
        for i in range(batch_size):
            # Insert target at random position
            candidates[i, :target_idx[i]] = distractors[i, :target_idx[i]]
            candidates[i, target_idx[i]] = target[i]
            candidates[i, target_idx[i]+1:] = distractors[i, target_idx[i]:]
        
        return {
            'target': target,
            'candidates': candidates,
            'target_idx': target_idx
        }
    
    def compute_similarity(self, prediction, candidates):
        """
        Compute similarity between Listener's prediction and all candidates.
        
        Args:
            prediction: Tensor of shape [batch_size, observation_dim]
            candidates: Tensor of shape [batch_size, num_candidates, observation_dim]
            
        Returns:
            Similarity scores of shape [batch_size, num_candidates]
        """
        # Expand prediction to match candidates shape
        prediction_expanded = tf.expand_dims(prediction, axis=1)
        
        # Compute cosine similarity
        # Normalize vectors
        prediction_norm = tf.nn.l2_normalize(prediction_expanded, axis=-1)
        candidates_norm = tf.nn.l2_normalize(candidates, axis=-1)
        
        # Dot product gives cosine similarity
        similarity = tf.reduce_sum(prediction_norm * candidates_norm, axis=-1)
        
        return similarity
    
    def compute_reward(self, prediction, candidates, target_idx):
        """
        Compute reward based on whether Listener chose the correct candidate.
        
        Args:
            prediction: Listener's prediction [batch_size, observation_dim]
            candidates: All candidates [batch_size, num_candidates, observation_dim]
            target_idx: Correct candidate indices [batch_size]
            
        Returns:
            Dictionary containing:
                - 'reward': Binary reward (1.0 for correct, 0.0 for incorrect)
                - 'accuracy': Same as reward (for metrics)
                - 'chosen_idx': Index of candidate Listener chose
                - 'similarities': Similarity scores for all candidates
        """
        # Compute similarities
        similarities = self.compute_similarity(prediction, candidates)
        
        # Listener chooses candidate with highest similarity
        chosen_idx = tf.argmax(similarities, axis=-1)
        
        # Reward is 1.0 if correct, 0.0 otherwise
        correct = tf.cast(tf.equal(chosen_idx, target_idx), tf.float32)
        
        return {
            'reward': correct,
            'accuracy': correct,
            'chosen_idx': chosen_idx,
            'similarities': similarities
        }
    
    def compute_loss(self, prediction, candidates, target_idx):
        """
        Compute cross-entropy loss for training.
        
        Args:
            prediction: Listener's prediction [batch_size, observation_dim]
            candidates: All candidates [batch_size, num_candidates, observation_dim]
            target_idx: Correct candidate indices [batch_size]
            
        Returns:
            Loss value
        """
        # Compute similarities (logits)
        similarities = self.compute_similarity(prediction, candidates)
        
        # Scale similarities to make them suitable as logits
        logits = similarities * SIMILARITY_TEMPERATURE
        
        # Cross-entropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            target_idx, logits, from_logits=True
        )
        
        return tf.reduce_mean(loss)


class ObservationGenerator:
    """
    Generates structured observations (e.g., colors, shapes).
    """
    
    def __init__(self, observation_type='random', num_attributes=3, seed=None):
        """
        Args:
            observation_type: Type of observations ('random', 'colors', 'shapes')
            num_attributes: Number of attributes per observation
            seed: Random seed
        """
        self.observation_type = observation_type
        self.num_attributes = num_attributes
        self.rng = np.random.RandomState(seed)
    
    def generate(self, batch_size=1):
        """Generate observations based on type."""
        if self.observation_type == 'random':
            # Random normalized vectors
            obs = self.rng.randn(batch_size, self.num_attributes).astype(np.float32)
            norms = np.linalg.norm(obs, axis=1, keepdims=True)
            return obs / (norms + 1e-8)
        
        elif self.observation_type == 'colors':
            # RGB color values
            return self.rng.rand(batch_size, 3).astype(np.float32)
        
        elif self.observation_type == 'shapes':
            # One-hot encoded shape attributes
            obs = np.zeros((batch_size, self.num_attributes), dtype=np.float32)
            for i in range(batch_size):
                for j in range(self.num_attributes):
                    obs[i, j] = self.rng.choice([0.0, 1.0])
            return obs
        
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")
