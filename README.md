# Neural Architecture Search & Optimization

An end-to-end Neural Architecture Search (NAS) system that uses an RNN controller and REINFORCE to automatically generate and evaluate CNN architectures on CIFAR-10.

The project also includes a Random Search baseline to compare learned architecture search against a simple non-learning strategy.

---

## Overview

Designing neural network architectures manually can require significant experimentation.

This project explores automated architecture design using Reinforcement Learning.

The system works as follows:

```text
RNN Controller
      ↓
Generate CNN Architecture
      ↓
Train Candidate Model
      ↓
Evaluate on Validation Set
      ↓
Validation Accuracy → Reward
      ↓
REINFORCE Update
      ↓
Generate Next Architecture
