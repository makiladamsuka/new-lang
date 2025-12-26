# Emergent Language Learning

A TensorFlow implementation where two AI agents develop their own language through communication.

## Overview

This project demonstrates how neural network agents can develop emergent communication protocols. Two agents (a Speaker and a Listener) start with no shared language and through repeated interactions in a referential game, they develop their own vocabulary and language patterns.

### How It Works

1. **The Game**: 
   - The Speaker observes a target object (represented as a vector)
   - The Speaker generates a message (sequence of discrete symbols) to describe it
   - The Listener receives the message and a set of candidate objects
   - The Listener must identify which candidate matches what the Speaker saw

2. **Learning**:
   - Both agents are neural networks trained end-to-end
   - They receive rewards when communication is successful
   - Through thousands of interactions, patterns emerge
   - Over time, consistent mappings form between objects and messages

3. **Key Innovation**:
   - Uses **Gumbel-Softmax** trick for differentiable discrete communication
   - Allows gradient-based optimization of discrete message generation
   - Enables end-to-end training of both agents simultaneously

## Project Structure

```
.
├── agents.py           # Speaker and Listener agent classes
├── models.py           # Neural network architectures
├── environment.py      # Referential game environment
├── train.py           # Main training script
├── utils.py           # Logging, visualization, and analysis tools
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/makiladamsuka/new-lang.git
cd new-lang
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run training with default parameters:

```bash
python train.py
```

This will:
- Train agents for 100 epochs
- Save logs to `logs/` directory
- Save checkpoints to `checkpoints/` directory
- Generate visualizations periodically

### Custom Configuration

Customize training with command-line arguments:

```bash
python train.py \
  --vocab-size 15 \
  --message-length 5 \
  --num-epochs 200 \
  --learning-rate 0.0005 \
  --observation-dim 20
```

### Key Parameters

**Environment:**
- `--observation-dim`: Dimensionality of objects (default: 10)
- `--num-candidates`: Number of candidate objects in the game (default: 4)

**Model:**
- `--vocab-size`: Number of discrete symbols available (default: 10)
- `--message-length`: Length of messages (default: 3)
- `--hidden-dim`: Size of hidden layers (default: 128)
- `--embedding-dim`: Size of embeddings (default: 64)

**Training:**
- `--num-epochs`: Number of training epochs (default: 100)
- `--steps-per-epoch`: Training steps per epoch (default: 100)
- `--learning-rate`: Learning rate (default: 0.001)
- `--initial-temperature`: Starting temperature for Gumbel-Softmax (default: 1.0)
- `--temperature-decay`: Temperature decay rate (default: 0.99)

**Logging:**
- `--log-dir`: Directory for logs (default: 'logs')
- `--checkpoint-dir`: Directory for checkpoints (default: 'checkpoints')
- `--eval-interval`: Evaluate every N epochs (default: 5)
- `--viz-interval`: Generate visualizations every N epochs (default: 10)

### Example Commands

**Quick test run:**
```bash
python train.py --num-epochs 20 --steps-per-epoch 50
```

**Larger vocabulary and longer messages:**
```bash
python train.py --vocab-size 20 --message-length 5 --num-epochs 150
```

**More difficult game (more candidates):**
```bash
python train.py --num-candidates 8 --num-epochs 200
```

## Understanding the Results

### Training Metrics

During training, you'll see:
- **Loss**: Cross-entropy loss (should decrease)
- **Accuracy**: Communication success rate (should increase to >80%)
- **Message Consistency**: How consistently same objects produce same messages
- **Message Entropy**: Diversity of messages used

### Expected Results

After training, agents should achieve:
- **>80% communication accuracy**: Successfully identify the correct object
- **High consistency**: Same objects reliably produce the same messages
- **Emergent structure**: Clear patterns in the language

### Visualizations

The training process generates several visualizations in the `logs/` directory:

1. **`training_curves.png`**: 
   - Shows accuracy, loss, and entropy over time
   - Track convergence and learning progress

2. **`message_distribution.png`**:
   - Displays most frequently used messages
   - Shows if agents use a diverse vocabulary

3. **`obs_msg_mapping.png`**:
   - Visualizes which observations produce which messages
   - Helps understand the emergent language patterns

### Example Output

```
Epoch 100 Summary:
  Average Loss: 0.1234
  Average Accuracy: 0.8567

Evaluating at epoch 100...
  Evaluation Accuracy: 0.8512
  Message Consistency: 0.9234
  Message Entropy: 2.1456
  Unique Mappings: 87
```

## Technical Details

### Architecture

**Speaker (Encoder + Message Generator):**
```
Observation → Encoder → Embedding → Message Generator → Message
                ↓                          ↓
           Dense Layers              Gumbel-Softmax
```

**Listener (Message Interpreter):**
```
Message → Flatten → Dense Layers → Prediction
```

### Gumbel-Softmax Trick

The key innovation enabling this system is the Gumbel-Softmax trick:

1. **Problem**: Discrete sampling is not differentiable
2. **Solution**: Add Gumbel noise and apply softmax with temperature
3. **Result**: Differentiable approximation of discrete sampling
4. **Benefit**: Can train end-to-end with gradient descent

Temperature parameter controls discreteness:
- High temperature (1.0+): Softer, more exploration
- Low temperature (0.5): Sharper, more discrete
- Temperature annealing: Start high, decay over time

### Training Algorithm

1. Sample a batch of referential games
2. Speaker encodes target and generates message
3. Listener interprets message and predicts target
4. Compute similarity between prediction and all candidates
5. Calculate cross-entropy loss based on correct candidate
6. Backpropagate and update both agents
7. Periodically evaluate and visualize

## Extending the Project

### Custom Observations

Modify `environment.py` to use different observation types:

```python
# Use color observations
env = ReferentialGameEnvironment(observation_dim=3)  # RGB colors

# Or create custom observation generator
from environment import ObservationGenerator
gen = ObservationGenerator(observation_type='colors', num_attributes=3)
```

### Different Network Architectures

Edit `models.py` to experiment with:
- Different encoder architectures (CNN, RNN, Transformer)
- Larger or smaller hidden dimensions
- Different activation functions

### Advanced Training Strategies

Try different training approaches:
- **REINFORCE**: Use policy gradient methods instead of cross-entropy
- **Curriculum Learning**: Start with easier games, increase difficulty
- **Multi-agent**: Add more than 2 agents
- **Compositional Messages**: Analyze if messages show compositionality

## Troubleshooting

### Low Accuracy (<50%)

- Increase training epochs: `--num-epochs 200`
- Adjust learning rate: `--learning-rate 0.0005`
- Simplify the task: `--num-candidates 2`
- Increase model capacity: `--hidden-dim 256`

### Messages Not Converging

- Decay temperature more aggressively: `--temperature-decay 0.95`
- Use lower minimum temperature: `--min-temperature 0.3`
- Check if vocabulary is large enough for the task

### Training Too Slow

- Reduce steps per epoch: `--steps-per-epoch 50`
- Reduce evaluation frequency: `--eval-interval 10`
- Use smaller models: `--hidden-dim 64 --embedding-dim 32`

## Research Background

This implementation is inspired by research on emergent communication:

- **Referential Games**: Lewis (1969), Clark (1996)
- **Neural Emergent Communication**: Lazaridou et al. (2016), Havrylov & Titov (2017)
- **Gumbel-Softmax**: Jang et al. (2016), Maddison et al. (2016)

Key papers:
- "Emergence of Language with Multi-agent Games" (Lazaridou et al., 2017)
- "Emergence of Linguistic Communication from Referential Games" (Havrylov & Titov, 2017)
- "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., 2016)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{emergent_language_2024,
  title = {Emergent Language Learning with TensorFlow},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/makiladamsuka/new-lang}
}
```

## Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Research community working on emergent communication
- Open source contributors

---

**Happy experimenting with emergent language!** 🗣️🤖
