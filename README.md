# Neural Architecture Search & Optimization

An end-to-end **reinforcement learning-based Neural Architecture Search (NAS)** system that uses an **LSTM controller and REINFORCE** to discover CNN architectures for CIFAR-10.

The project compares learned architecture search with a **Random Search baseline** under the same search budget while explicitly balancing validation performance and model efficiency.

## Highlights

- LSTM-based architecture controller
- REINFORCE policy-gradient optimization
- Automated CNN architecture generation
- Configurable architecture search space
- Accuracy + parameter-efficiency reward
- Random Search baseline with matched budget
- Train / validation / test separation
- PyTorch + CUDA support
- Modular search, training, and evaluation pipeline

## System Workflow

```text
CIFAR-10
   |
   v
Search Space
   |
   v
LSTM Controller
   |
   v
Sample Architecture
   |
   v
Train Candidate CNN
   |
   v
Validation Accuracy + Parameter Cost
   |
   v
Efficiency-Aware Reward
   |
   v
REINFORCE Update
   |
   +-------> Next Architecture

Selected Architecture
   |
   v
Retrain From Scratch
   |
   v
Held-Out Test Evaluation
```

## Search Space

The controller selects four architectural decisions:

| Parameter | Options |
| --- | --- |
| Filters | 16, 32, 64 |
| Kernel size | 3x3, 5x5 |
| Pooling | Max, Average |
| Activation | ReLU, GELU |

This gives **3 x 2 x 2 x 2 = 24 possible architectures**.

Each generated CNN follows:

```text
Input
  -> Convolution
  -> Batch Normalization
  -> Activation
  -> Pooling
  -> Convolution
  -> Batch Normalization
  -> Activation
  -> Pooling
  -> Global Average Pooling
  -> Fully Connected Layer
  -> 10-Class Output
```

## Reinforcement Learning Objective

For every sampled architecture:

1. Construct the CNN.
2. Train it on the training split.
3. Measure validation accuracy.
4. Measure trainable parameters.
5. Compute an efficiency-aware reward.
6. Update the controller with REINFORCE.

The current reward is conceptually:

```text
Reward = normalized validation accuracy
       - lambda * normalized parameter cost
```

where `lambda` controls the importance of model compactness.

## Experiment Configuration

| Setting | Value |
| --- | ---: |
| Search budget | 20 architectures |
| Candidate training | 3 epochs |
| Parameter penalty | 0.1 |
| Final training | 10 epochs |

RL-NAS and Random Search use the same search budget and candidate-training configuration for a direct comparison.

## Results

### RL-NAS

Best discovered architecture:

```text
Filters:      32
Kernel size:  5x5
Pooling:      Average Pooling
Activation:   GELU
```

| Metric | Result |
| --- | ---: |
| Validation accuracy | 52.08% |
| Parameters | 54,538 |
| Reward | 0.5026 |

### Random Search

Best discovered architecture:

```text
Filters:      64
Kernel size:  3x3
Pooling:      Max Pooling
Activation:   GELU
```

| Metric | Result |
| --- | ---: |
| Validation accuracy | 54.38% |
| Parameters | 77,322 |
| Reward | 0.5180 |

### Comparison

| Metric | RL-NAS | Random Search |
| --- | ---: | ---: |
| Search budget | 20 | 20 |
| Candidate epochs | 3 | 3 |
| Best validation accuracy | 52.08% | **54.38%** |
| Parameters | **54,538** | 77,322 |
| Best reward | 0.5026 | **0.5180** |

Random Search achieved the strongest validation accuracy and reward in this experiment. RL-NAS, however, found a competitive architecture with approximately **29.5% fewer trainable parameters** than the best Random Search architecture.

The result is therefore presented as a performance-efficiency trade-off, not as evidence that reinforcement learning always beats random search.

## Final Evaluation

The selected RL-NAS architecture was retrained from scratch before final evaluation. The CIFAR-10 test set was kept out of architecture selection.

```text
Selected architecture:
32 filters | 5x5 kernel | Average Pooling | GELU

Parameters: 54,538
Training epochs: 10
Validation accuracy: 62.28%
Test accuracy: 61.90%
```

### Evaluation protocol

```text
Architecture Search
        |
        v
Select RL-NAS Architecture
        |
        v
Retrain From Scratch
        |
        v
Validation Evaluation
        |
        v
Held-Out Test Evaluation
```

This keeps the test set from influencing architecture selection.

## Project Structure

```text
Neural-Architecture-Search-Optimization/
|
|-- src/
|   |-- controller.py       # LSTM architecture controller
|   |-- model.py            # Dynamic CNN model
|   |-- nas.py              # NAS utilities
|   |-- reinforce.py        # REINFORCE optimization
|   |-- results.py          # Experiment result utilities
|   |-- search_space.py     # Architecture search space
|   `-- trainer.py          # CIFAR-10 training/evaluation
|
|-- search.py               # RL-based NAS experiment
|-- baseline.py             # Random Search baseline
|-- final_train.py          # Final training/evaluation
|-- requirements.txt        # Python dependencies
|-- .gitignore
`-- README.md
```

## Installation

```bash
git clone https://github.com/Atul1127/Neural-Architecture-Search-Optimization.git
cd Neural-Architecture-Search-Optimization
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Windows
.venv\\Scripts\\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for architecture search.

## Usage

Run the RL-based NAS experiment:

```bash
python search.py
```

Run the matched Random Search baseline:

```bash
python baseline.py
```

Retrain and evaluate the selected architecture:

```bash
python final_train.py
```

## Technologies

Python | PyTorch | Torchvision | Reinforcement Learning | REINFORCE | LSTM/RNN | CNN | CIFAR-10 | CUDA

## Project Goal

Demonstrate how reinforcement learning can automate neural network architecture design while explicitly considering the trade-off between predictive performance and model efficiency.
