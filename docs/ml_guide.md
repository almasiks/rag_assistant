# Machine Learning Guide

## Introduction
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

## Supervised Learning
In supervised learning, models are trained on labeled data. The algorithm learns to map input features to output labels.
Common algorithms include linear regression, logistic regression, decision trees, and neural networks.
Applications include image classification, spam detection, and price prediction.

## Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data.
Common techniques include clustering (K-Means, DBSCAN), dimensionality reduction (PCA, t-SNE), and autoencoders.
Applications include customer segmentation, anomaly detection, and recommendation systems.

## Reinforcement Learning
Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones.
Key concepts include reward functions, policies, and value functions.
Applications include game playing (AlphaGo), robotics, and autonomous driving.

## Model Evaluation
Evaluating machine learning models requires proper metrics.
For classification: accuracy, precision, recall, F1-score, ROC-AUC.
For regression: MAE, MSE, RMSE, R-squared.
Cross-validation is essential to avoid overfitting and ensure generalization.

## Regularization
Regularization prevents overfitting by adding a penalty to the loss function.
L1 regularization (Lasso) produces sparse models by pushing weights to zero.
L2 regularization (Ridge) shrinks weights evenly and handles correlated features well.
ElasticNet combines both L1 and L2 penalties.

## Neural Networks
Neural networks consist of layers of interconnected neurons.
Deep learning uses many layers to learn hierarchical representations.
Common architectures include CNNs for images, RNNs for sequences, and Transformers for text.
Training uses backpropagation and gradient descent optimization.
