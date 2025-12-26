"""
Utility functions for logging, visualization, and analysis.

This module provides tools to:
- Log training metrics
- Visualize emerging language patterns
- Analyze what messages mean
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import tensorflow as tf


class Logger:
    """
    Logger for training metrics and language analysis.
    """
    
    def __init__(self, log_dir='logs'):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.message_history = []
    
    def log_metrics(self, step, **kwargs):
        """
        Log scalar metrics.
        
        Args:
            step: Training step
            **kwargs: Metric name-value pairs
        """
        for name, value in kwargs.items():
            self.metrics[name].append({'step': step, 'value': float(value)})
    
    def log_message(self, observation, message, target_idx, chosen_idx, correct):
        """
        Log a communication instance.
        
        Args:
            observation: The target observation
            message: The generated message
            target_idx: Index of target candidate
            chosen_idx: Index chosen by Listener
            correct: Whether communication was successful
        """
        self.message_history.append({
            'observation': observation.tolist() if hasattr(observation, 'tolist') else observation,
            'message': message.tolist() if hasattr(message, 'tolist') else message,
            'target_idx': int(target_idx),
            'chosen_idx': int(chosen_idx),
            'correct': bool(correct)
        })
    
    def save_metrics(self, filename='metrics.json'):
        """Save metrics to JSON file."""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
    
    def save_messages(self, filename='messages.json'):
        """Save message history to JSON file."""
        filepath = os.path.join(self.log_dir, filename)
        # Save only recent messages to avoid huge files
        recent_messages = self.message_history[-1000:]
        with open(filepath, 'w') as f:
            json.dump(recent_messages, f, indent=2)
    
    def plot_metrics(self, metrics_to_plot=None, filename='training_curves.png'):
        """
        Plot training metrics.
        
        Args:
            metrics_to_plot: List of metric names to plot (None = all)
            filename: Output filename
        """
        if metrics_to_plot is None:
            metrics_to_plot = list(self.metrics.keys())
        
        num_metrics = len(metrics_to_plot)
        if num_metrics == 0:
            return
        
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
        if num_metrics == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, metrics_to_plot):
            if metric_name not in self.metrics:
                continue
            
            data = self.metrics[metric_name]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            
            ax.plot(steps, values)
            ax.set_xlabel('Step')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over time')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.log_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()


class LanguageAnalyzer:
    """
    Analyze the emergent language to understand message meanings.
    """
    
    def __init__(self, vocab_size, message_length):
        """
        Args:
            vocab_size: Size of vocabulary
            message_length: Length of messages
        """
        self.vocab_size = vocab_size
        self.message_length = message_length
    
    def message_to_string(self, message):
        """
        Convert a message tensor to a string representation.
        
        Args:
            message: Tensor of shape [message_length, vocab_size] or [batch, message_length, vocab_size]
            
        Returns:
            String representation of the message
        """
        if len(message.shape) == 3:
            # Batch of messages, take first one
            message = message[0]
        
        # Convert one-hot to symbol indices
        symbols = tf.argmax(message, axis=-1).numpy()
        return ''.join([chr(ord('A') + s) for s in symbols])
    
    def analyze_consistency(self, observations, messages):
        """
        Analyze whether same observations produce same messages.
        
        Args:
            observations: Array of observations
            messages: Array of corresponding messages
            
        Returns:
            Dictionary with consistency metrics
        """
        message_map = defaultdict(list)
        
        for obs, msg in zip(observations, messages):
            msg_str = self.message_to_string(msg)
            obs_tuple = tuple(obs.flatten())
            message_map[obs_tuple].append(msg_str)
        
        # Calculate consistency: how often does same observation -> same message?
        consistencies = []
        for obs_tuple, msg_list in message_map.items():
            if len(msg_list) > 1:
                # Most common message
                counter = Counter(msg_list)
                most_common_count = counter.most_common(1)[0][1]
                consistency = most_common_count / len(msg_list)
                consistencies.append(consistency)
        
        return {
            'mean_consistency': np.mean(consistencies) if consistencies else 0.0,
            'unique_mappings': len(message_map)
        }
    
    def compute_message_entropy(self, messages):
        """
        Compute entropy of message distribution.
        
        Args:
            messages: Array of messages
            
        Returns:
            Entropy value
        """
        message_strings = [self.message_to_string(msg) for msg in messages]
        
        counter = Counter(message_strings)
        total = len(message_strings)
        
        entropy = 0.0
        for count in counter.values():
            p = count / total
            entropy -= p * np.log(p + 1e-10)
        
        return entropy
    
    def plot_message_distribution(self, messages, top_k=20, filename='message_distribution.png', log_dir='logs'):
        """
        Plot distribution of most common messages.
        
        Args:
            messages: Array of messages
            top_k: Number of top messages to show
            filename: Output filename
            log_dir: Directory to save plot
        """
        message_strings = [self.message_to_string(msg) for msg in messages]
        
        counter = Counter(message_strings)
        most_common = counter.most_common(top_k)
        
        if not most_common:
            return
        
        labels, counts = zip(*most_common)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.xlabel('Message')
        plt.ylabel('Frequency')
        plt.title(f'Top {top_k} Most Common Messages')
        plt.tight_layout()
        
        filepath = os.path.join(log_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_observation_message_mapping(self, observations, messages, 
                                              max_samples=50, filename='obs_msg_mapping.png',
                                              log_dir='logs'):
        """
        Visualize which observations produce which messages.
        
        Args:
            observations: Array of observations
            messages: Array of corresponding messages
            max_samples: Maximum number of samples to visualize
            filename: Output filename
            log_dir: Directory to save plot
        """
        # Limit samples for visualization
        n_samples = min(len(observations), max_samples)
        obs_sample = observations[:n_samples]
        msg_sample = messages[:n_samples]
        
        # Convert messages to strings
        msg_strings = [self.message_to_string(msg) for msg in msg_sample]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot observations as heatmap
        ax1.imshow(obs_sample.T, aspect='auto', cmap='viridis')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Observation Dimension')
        ax1.set_title('Observations')
        
        # Plot messages as text
        ax2.axis('off')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, n_samples)
        
        for i, msg in enumerate(msg_strings):
            ax2.text(0, n_samples - i - 0.5, f"{i}: {msg}", fontsize=8, verticalalignment='center')
        
        ax2.set_title('Generated Messages')
        
        plt.tight_layout()
        filepath = os.path.join(log_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()


def create_checkpoint_manager(agents, checkpoint_dir='checkpoints', max_to_keep=5):
    """
    Create a checkpoint manager for saving model weights.
    
    Args:
        agents: CommunicationAgents instance
        checkpoint_dir: Directory to save checkpoints
        max_to_keep: Maximum number of checkpoints to keep
        
    Returns:
        Checkpoint manager object
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return {
        'checkpoint_dir': checkpoint_dir,
        'agents': agents,
        'max_to_keep': max_to_keep,
        'checkpoint_count': 0
    }


def save_checkpoint(checkpoint_manager, step):
    """
    Save a checkpoint.
    
    Args:
        checkpoint_manager: Manager object from create_checkpoint_manager
        step: Current training step
    """
    checkpoint_dir = checkpoint_manager['checkpoint_dir']
    agents = checkpoint_manager['agents']
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{step}')
    agents.save_weights(checkpoint_path)
    
    # Save metadata
    metadata_path = os.path.join(checkpoint_dir, f'checkpoint_{step}_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({'step': step}, f)
    
    print(f"Checkpoint saved at step {step}")


def load_checkpoint(checkpoint_manager, checkpoint_path):
    """
    Load a checkpoint.
    
    Args:
        checkpoint_manager: Manager object
        checkpoint_path: Path to checkpoint (without extension)
    """
    agents = checkpoint_manager['agents']
    agents.load_weights(checkpoint_path)
    print(f"Checkpoint loaded from {checkpoint_path}")
