
# Neural Architecture Search & Optimization

An end-to-end **Reinforcement Learning-based Neural Architecture Search (NAS)** system that uses an **LSTM controller and REINFORCE** to automatically discover CNN architectures for CIFAR-10.

The project compares learned architecture search against a **Random Search baseline** under the same search budget while optimizing for both validation performance and model efficiency.

---

## Overview

Instead of manually designing CNN architectures, the system learns to generate promising architectures through reinforcement learning.

```text
                    CIFAR-10
                       │
                       ▼
                Search Space
                       │
                       ▼
                LSTM Controller
                       │
                       ▼
             Sample Architecture
                       │
                       ▼
                 Train CNN
                       │
                       ▼
              Validation Accuracy
                       │
                       ▼
          Accuracy + Efficiency Reward
                       │
                       ▼
                  REINFORCE
                       │
                       ▼
             Update Controller
                       │
                       └──────────► Next Architecture
````

After architecture search, the selected architecture is retrained from scratch and evaluated on the held-out CIFAR-10 test set.

---

## Key Features

* LSTM-based neural architecture controller
* REINFORCE policy-gradient optimization
* Automated CNN architecture generation
* Configurable CNN search space
* Accuracy and parameter-efficiency reward
* Random Search baseline
* Equal search budget for fair comparison
* Train / validation / test separation
* GPU acceleration with PyTorch
* Modular implementation
* Separate architecture search and final evaluation

---

## Architecture Search Space

The controller searches over four architectural decisions:

| Parameter   | Options                      |
| ----------- | ---------------------------- |
| Filters     | 16, 32, 64                   |
| Kernel Size | 3×3, 5×5                     |
| Pooling     | Max Pooling, Average Pooling |
| Activation  | ReLU, GELU                   |

This produces:

```text
3 × 2 × 2 × 2 = 24 possible architectures
```

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

---

## Reinforcement Learning

The LSTM controller generates architectural decisions by sampling from categorical probability distributions.

For each sampled architecture:

1. A CNN is constructed.
2. The candidate is trained on the training set.
3. Validation accuracy is measured.
4. Trainable parameter count is measured.
5. An efficiency-aware reward is calculated.
6. REINFORCE updates the controller.

### Reward

The current objective combines validation accuracy with a parameter-efficiency penalty:

```text
Reward =
    normalized validation accuracy
    − λ × normalized parameter cost
```

where `λ` controls the importance of model compactness.

This allows the search to consider both predictive performance and model size.

---

## Search Configuration

The current experiment evaluates:

```text
Search Budget:       20 architectures
Candidate Training:  3 epochs
Parameter Penalty:   0.1
```

Both RL-NAS and Random Search use the same search budget and candidate-training configuration.

---

# Experiments

## RL-Based NAS

The RL controller searched 20 candidate architectures using an LSTM controller and REINFORCE.

### Best Discovered Architecture

```text
Filters:       32
Kernel Size:   5×5
Pooling:       Average Pooling
Activation:    GELU
```

### Search Result

```text
Validation Accuracy: 52.08%
Parameters:          54,538
Reward:              0.5026
```

---

## Random Search Baseline

Random Search independently evaluated 20 candidate architectures using the same training budget and efficiency-aware objective.

### Best Discovered Architecture

```text
Filters:       64
Kernel Size:   3×3
Pooling:       Max Pooling
Activation:    GELU
```

### Search Result

```text
Validation Accuracy: 54.38%
Parameters:          77,322
Reward:              0.5180
```

---

## RL-NAS vs Random Search

| Metric                   |     RL-NAS | Random Search |
| ------------------------ | ---------: | ------------: |
| Search Budget            |         20 |            20 |
| Candidate Epochs         |          3 |             3 |
| Best Validation Accuracy |     52.08% |    **54.38%** |
| Parameters               | **54,538** |        77,322 |
| Best Reward              |     0.5026 |    **0.5180** |

### Interpretation

Random Search achieved the highest validation accuracy and reward in the current search experiment.

However, RL-NAS discovered a competitive architecture with approximately **29.5% fewer trainable parameters** than the best Random Search architecture.

The experiment therefore demonstrates the trade-off between predictive performance and model efficiency rather than claiming that reinforcement learning always outperforms random search.

---

# Final Evaluation

After architecture search, the best RL-NAS architecture was selected and retrained from scratch using a larger training budget.

The CIFAR-10 test set was kept separate from architecture selection and used only for final evaluation.

### Selected Architecture

```text
Filters:       32
Kernel Size:   5×5
Pooling:       Average Pooling
Activation:    GELU
Parameters:    54,538
```

### Final Training

```text
Training Epochs: 10
```

### Final Results

```text
Validation Accuracy: 62.28%
Test Accuracy:       61.90%
Trainable Parameters: 54,538
```

The final test accuracy of **61.90%** was obtained on the held-out CIFAR-10 test set after the architecture had already been selected using the training/validation data.

### Evaluation Workflow

```text
Architecture Search
        ↓
Select RL-NAS Architecture
        ↓
Retrain From Scratch
        ↓
Validation Evaluation
        ↓
Held-Out Test Evaluation
```

This prevents the test set from influencing architecture selection.

---

## Project Structure

```text
Neural-Architecture-Search-Optimization/
│
├── src/
│   ├── controller.py       # LSTM architecture controller
│   ├── model.py            # Dynamic CNN model
│   ├── nas.py              # NAS utilities
│   ├── reinforce.py        # REINFORCE optimization
│   ├── results.py          # Experiment result utilities
│   ├── search_space.py     # Architecture search space
│   └── trainer.py          # CIFAR-10 training/evaluation
│
├── search.py               # RL-based NAS experiment
├── baseline.py             # Random Search baseline
├── final_train.py          # Final architecture training/evaluation
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

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

A CUDA-enabled GPU is recommended for running the architecture search efficiently.

---

## Usage

### Run RL-Based NAS

```bash
python search.py
```

### Run Random Search Baseline

```bash
python baseline.py
```

### Train and Evaluate the Selected Architecture

```bash
python final_train.py
```

---

## Technologies

* Python
* PyTorch
* Torchvision
* Reinforcement Learning
* REINFORCE
* LSTM / RNN
* Convolutional Neural Networks
* CIFAR-10
* CUDA

---

## Project Goal

The goal of this project is to demonstrate how reinforcement learning can be applied to **automated neural network architecture design**, while explicitly considering the trade-off between model performance and model efficiency.

