"""
Main training script for emergent language learning.

This script:
1. Initializes agents and environment
2. Runs training episodes where agents communicate
3. Updates agent weights based on communication success
4. Logs metrics and saves checkpoints
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from agents import CommunicationAgents
from environment import ReferentialGameEnvironment
from utils import Logger, LanguageAnalyzer, create_checkpoint_manager, save_checkpoint


def train_step(agents, environment, optimizer_speaker, optimizer_listener):
    """
    Execute one training step.
    
    Args:
        agents: CommunicationAgents instance
        environment: ReferentialGameEnvironment instance
        optimizer_speaker: Optimizer for Speaker
        optimizer_listener: Optimizer for Listener
        
    Returns:
        Dictionary of metrics
    """
    # Create a game instance
    game = environment.create_game(batch_size=32)
    
    target = tf.constant(game['target'])
    candidates = tf.constant(game['candidates'])
    target_idx = tf.constant(game['target_idx'])
    
    # Forward pass with gradient tracking
    with tf.GradientTape(persistent=True) as tape:
        # Speaker generates message from target observation
        message = agents.speaker.get_message(target, training=True)
        
        # Listener interprets message to get prediction (embedding)
        prediction = agents.listener.interpret_message(message)
        
        # Encode all candidates to embeddings for comparison
        batch_size = tf.shape(candidates)[0]
        num_candidates = tf.shape(candidates)[1]
        # Reshape candidates from [batch, num_candidates, obs_dim] to [batch*num_candidates, obs_dim]
        candidates_flat = tf.reshape(candidates, [-1, tf.shape(candidates)[-1]])
        # Encode candidates
        candidates_encoded = agents.speaker.encoder(candidates_flat)
        # Reshape back to [batch, num_candidates, embedding_dim]
        candidates_embedded = tf.reshape(
            candidates_encoded, 
            [batch_size, num_candidates, tf.shape(candidates_encoded)[-1]]
        )
        
        # Compute loss using embeddings
        loss = environment.compute_loss(prediction, candidates_embedded, target_idx)
    
    # Compute gradients and update
    speaker_vars = agents.speaker.get_trainable_variables()
    listener_vars = agents.listener.get_trainable_variables()
    
    speaker_grads = tape.gradient(loss, speaker_vars)
    listener_grads = tape.gradient(loss, listener_vars)
    
    optimizer_speaker.apply_gradients(zip(speaker_grads, speaker_vars))
    optimizer_listener.apply_gradients(zip(listener_grads, listener_vars))
    
    del tape
    
    # Compute metrics
    reward_info = environment.compute_reward(prediction, candidates_embedded, target_idx)
    accuracy = tf.reduce_mean(reward_info['accuracy'])
    
    return {
        'loss': loss.numpy(),
        'accuracy': accuracy.numpy(),
        'message': message.numpy()
    }


def evaluate(agents, environment, num_episodes=100):
    """
    Evaluate agents without training.
    
    Args:
        agents: CommunicationAgents instance
        environment: ReferentialGameEnvironment instance
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of evaluation metrics
    """
    accuracies = []
    all_observations = []
    all_messages = []
    
    for _ in range(num_episodes):
        game = environment.create_game(batch_size=1)
        
        target = tf.constant(game['target'])
        candidates = tf.constant(game['candidates'])
        target_idx = game['target_idx']
        
        # Generate message (no training)
        message = agents.speaker.get_message(target, training=False)
        prediction = agents.listener.interpret_message(message)
        
        # Encode candidates for comparison
        candidates_flat = tf.reshape(candidates, [-1, tf.shape(candidates)[-1]])
        candidates_encoded = agents.speaker.encoder(candidates_flat)
        candidates_embedded = tf.reshape(
            candidates_encoded,
            [1, tf.shape(candidates)[1], tf.shape(candidates_encoded)[-1]]
        )
        
        # Compute accuracy
        reward_info = environment.compute_reward(prediction, candidates_embedded, target_idx)
        accuracies.append(reward_info['accuracy'].numpy()[0])
        
        all_observations.append(target.numpy()[0])
        all_messages.append(message.numpy())
    
    return {
        'accuracy': np.mean(accuracies),
        'observations': np.array(all_observations),
        'messages': np.array(all_messages)
    }


def train(config):
    """
    Main training loop.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 60)
    print("Emergent Language Learning - Training")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Initialize environment
    env = ReferentialGameEnvironment(
        observation_dim=config['observation_dim'],
        num_candidates=config['num_candidates'],
        seed=config['seed']
    )
    
    # Initialize agents
    agents = CommunicationAgents(
        observation_dim=config['observation_dim'],
        vocab_size=config['vocab_size'],
        message_length=config['message_length'],
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        temperature=config['initial_temperature'],
        learning_rate=config['learning_rate']
    )
    
    # Optimizers (already created in agents, but we can also create separate ones)
    optimizer_speaker = agents.speaker.optimizer
    optimizer_listener = agents.listener.optimizer
    
    # Initialize logging
    logger = Logger(log_dir=config['log_dir'])
    analyzer = LanguageAnalyzer(config['vocab_size'], config['message_length'])
    
    # Checkpoint manager
    checkpoint_manager = create_checkpoint_manager(
        agents, 
        checkpoint_dir=config['checkpoint_dir'],
        max_to_keep=config['max_checkpoints']
    )
    
    print("\nStarting training...")
    print("-" * 60)
    
    # Training loop
    step = 0
    for epoch in range(config['num_epochs']):
        epoch_losses = []
        epoch_accuracies = []
        
        # Progress bar for steps in this epoch
        steps_per_epoch = config['steps_per_epoch']
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for _ in pbar:
            # Train step
            metrics = train_step(agents, env, optimizer_speaker, optimizer_listener)
            
            epoch_losses.append(metrics['loss'])
            epoch_accuracies.append(metrics['accuracy'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}"
            })
            
            step += 1
            
            # Log metrics periodically
            if step % config['log_interval'] == 0:
                logger.log_metrics(
                    step=step,
                    loss=metrics['loss'],
                    accuracy=metrics['accuracy']
                )
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        
        # Evaluate
        if (epoch + 1) % config['eval_interval'] == 0:
            print(f"\nEvaluating at epoch {epoch+1}...")
            eval_results = evaluate(agents, env, num_episodes=config['eval_episodes'])
            
            print(f"  Evaluation Accuracy: {eval_results['accuracy']:.4f}")
            
            # Analyze language
            consistency = analyzer.analyze_consistency(
                eval_results['observations'],
                eval_results['messages']
            )
            entropy = analyzer.compute_message_entropy(eval_results['messages'])
            
            print(f"  Message Consistency: {consistency['mean_consistency']:.4f}")
            print(f"  Message Entropy: {entropy:.4f}")
            print(f"  Unique Mappings: {consistency['unique_mappings']}")
            
            # Log evaluation metrics
            logger.log_metrics(
                step=step,
                eval_accuracy=eval_results['accuracy'],
                message_consistency=consistency['mean_consistency'],
                message_entropy=entropy
            )
            
            # Visualizations
            if (epoch + 1) % config['viz_interval'] == 0:
                print("  Generating visualizations...")
                analyzer.plot_message_distribution(
                    eval_results['messages'],
                    top_k=20,
                    log_dir=config['log_dir']
                )
                analyzer.visualize_observation_message_mapping(
                    eval_results['observations'],
                    eval_results['messages'],
                    max_samples=50,
                    log_dir=config['log_dir']
                )
        
        # Update temperature (anneal for more discrete messages over time)
        if config['temperature_decay'] < 1.0:
            new_temp = config['initial_temperature'] * (config['temperature_decay'] ** (epoch + 1))
            new_temp = max(new_temp, config['min_temperature'])
            agents.set_temperature(new_temp)
            if (epoch + 1) % 10 == 0:
                print(f"  Temperature: {new_temp:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            save_checkpoint(checkpoint_manager, step)
        
        print("-" * 60)
    
    # Final evaluation
    print("\nFinal Evaluation...")
    final_results = evaluate(agents, env, num_episodes=1000)
    print(f"Final Accuracy: {final_results['accuracy']:.4f}")
    
    # Save final results
    logger.save_metrics()
    logger.save_messages()
    logger.plot_metrics(
        metrics_to_plot=['accuracy', 'loss', 'eval_accuracy', 'message_entropy'],
        filename='training_curves.png'
    )
    
    # Final checkpoint
    save_checkpoint(checkpoint_manager, step)
    
    print("\nTraining complete!")
    print("=" * 60)


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train emergent language agents')
    
    # Environment parameters
    parser.add_argument('--observation-dim', type=int, default=10,
                      help='Dimensionality of observations')
    parser.add_argument('--num-candidates', type=int, default=4,
                      help='Number of candidates in referential game')
    
    # Model parameters
    parser.add_argument('--vocab-size', type=int, default=10,
                      help='Size of vocabulary')
    parser.add_argument('--message-length', type=int, default=3,
                      help='Length of messages')
    parser.add_argument('--hidden-dim', type=int, default=128,
                      help='Size of hidden layers')
    parser.add_argument('--embedding-dim', type=int, default=64,
                      help='Size of embeddings')
    
    # Training parameters
    parser.add_argument('--num-epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                      help='Number of steps per epoch')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--initial-temperature', type=float, default=1.0,
                      help='Initial Gumbel-Softmax temperature')
    parser.add_argument('--temperature-decay', type=float, default=0.99,
                      help='Temperature decay rate per epoch')
    parser.add_argument('--min-temperature', type=float, default=0.5,
                      help='Minimum temperature')
    
    # Logging and checkpointing
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Directory for checkpoints')
    parser.add_argument('--log-interval', type=int, default=10,
                      help='Log metrics every N steps')
    parser.add_argument('--eval-interval', type=int, default=5,
                      help='Evaluate every N epochs')
    parser.add_argument('--eval-episodes', type=int, default=100,
                      help='Number of episodes for evaluation')
    parser.add_argument('--checkpoint-interval', type=int, default=20,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--max-checkpoints', type=int, default=5,
                      help='Maximum number of checkpoints to keep')
    parser.add_argument('--viz-interval', type=int, default=10,
                      help='Generate visualizations every N epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create config dictionary
    config = vars(args)
    
    # Run training
    train(config)


if __name__ == '__main__':
    main()
