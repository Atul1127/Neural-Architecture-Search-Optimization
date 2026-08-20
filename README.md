
# Neural Architecture Search & Optimization

An end-to-end Neural Architecture Search (NAS) system that uses an RNN controller and REINFORCE to automatically generate and evaluate CNN architectures on CIFAR-10.

The project also compares the RL-based search with a Random Search baseline.

## Overview

Instead of manually designing CNN architectures, this project uses Reinforcement Learning to automatically search for promising architectures.

```text
RNN Controller
      ↓
Generate CNN Architecture
      ↓
Train Candidate Model
      ↓
Validation Accuracy
      ↓
Reward
      ↓
REINFORCE Update
      ↓
Generate Next Architecture
````

After the search phase, the best architecture is trained for longer and evaluated on the held-out CIFAR-10 test set.

## Key Features

* RNN/LSTM-based architecture controller
* REINFORCE policy-gradient optimization
* Automated CNN architecture generation
* CIFAR-10 dataset
* Configurable architecture search space
* Random Search baseline
* Train / validation / test separation
* GPU support through PyTorch
* Modular implementation
* Independent final test evaluation

## Architecture Search Space

The controller searches over four architectural decisions:

| Parameter   | Options                      |
| ----------- | ---------------------------- |
| Filters     | 16, 32, 64                   |
| Kernel Size | 3×3, 5×5                     |
| Pooling     | Max Pooling, Average Pooling |
| Activation  | ReLU, GELU                   |

The generated CNN follows:

```text
Input
  ↓
Convolution
  ↓
Batch Normalization
  ↓
Activation
  ↓
Pooling
  ↓
Convolution
  ↓
Batch Normalization
  ↓
Activation
  ↓
Pooling
  ↓
Global Average Pooling
  ↓
Fully Connected Layer
  ↓
10-Class Output
```

## Reinforcement Learning

The RNN controller generates architectural decisions by sampling from categorical probability distributions.

The validation accuracy of each candidate architecture is used as the reward.

The controller is optimized using the REINFORCE objective:

```text
Loss = -log_probability × advantage
```

where:

```text
advantage = reward - baseline
```

A moving-average reward baseline is used to reduce policy-gradient variance.

## Experiments

### RL-Based NAS

Search budget: **5 architectures**

Best validation accuracy:

```text
47.90%
```

Best architecture:

```text
Filters:      64
Kernel Size:  5×5
Pooling:      Max Pooling
Activation:   GELU
```

### Random Search Baseline

Search budget: **5 architectures**

Best validation accuracy:

```text
48.02%
```

Best architecture:

```text
Filters:      64
Kernel Size:  5×5
Pooling:      Max Pooling
Activation:   GELU
```

The RL-based search produced a competitive architecture under the same limited search budget.

## Final Evaluation

The best discovered architecture was trained for **10 epochs** and evaluated on the untouched CIFAR-10 test set.

```text
Final Test Accuracy: 61.66%
```

The test set was not used during architecture selection.

## Project Structure

```text
Neural-Architecture-Search-Optimization/
│
├── src/
│   ├── controller.py       # RNN architecture controller
│   ├── model.py            # Dynamic CNN model
│   ├── nas.py              # Random Search implementation
│   ├── reinforce.py        # REINFORCE optimization
│   ├── results.py          # Experiment result utilities
│   ├── search_space.py     # Architecture search space
│   └── trainer.py          # CIFAR-10 training/evaluation
│
├── baseline.py             # Random Search experiment
├── search.py               # RL-based NAS experiment
├── final_train.py          # Final architecture training
├── test_model.py           # Model testing
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Atul1127/Neural-Architecture-Search-Optimization.git
cd Neural-Architecture-Search-Optimization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run RL-Based NAS

```bash
python search.py
```

### Run Random Search Baseline

```bash
python baseline.py
```

### Train the Best Architecture

```bash
python final_train.py
```

For faster experimentation, a CUDA-enabled GPU is recommended.

## Technologies

* Python
* PyTorch
* Torchvision
* Reinforcement Learning
* REINFORCE
* RNN / LSTM
* Convolutional Neural Networks
* CIFAR-10

