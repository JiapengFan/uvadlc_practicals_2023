  ################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
from train_mlp_numpy import confusion_matrix as confusion_matrix_np, confusion_matrix_to_metrics as confusion_matrix_to_metrics_np
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    conf_mat = confusion_matrix_np(predictions, targets)
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    metrics = confusion_matrix_to_metrics_np(confusion_matrix, beta)
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    betas = [0.1, 1, 10]
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_prob = []
    data_y = []
    for imgs, labels in data_loader:
      x, y = imgs.to(device), labels.to(device)
      with torch.no_grad():
          preds = model(x)
          data_y.extend(y.tolist())
          data_prob.extend(preds.tolist())
    
    conf_mat = confusion_matrix(np.array(data_prob), np.array(data_y))
    for beta in betas:
      metrics = confusion_matrix_to_metrics(conf_mat, beta)
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    # TODO: Initialize model and loss module
    n_inputs = np.array(cifar10['train'][0][0].shape).prod()
    model = MLP(n_inputs, hidden_dims, 10, use_batch_norm)
    loss_module = nn.CrossEntropyLoss()

    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer
    logging_dict = {'loss_train': [], 'accuracy_train': [], 'loss_val': [], 'accuracy_val': []}

    n_samples_train = len(cifar10['train'])
    n_samples_validation = len(cifar10['validation'])
    num_batches_train = int(np.ceil(n_samples_train/batch_size))
    num_batches_val = int(np.ceil(n_samples_validation/batch_size))

    weights_train = np.array([batch_size] * (num_batches_train - 1) + [n_samples_train % batch_size or batch_size])
    weights_train_sum = weights_train.sum()
    weights_val = np.array([batch_size] * (num_batches_val - 1) + [n_samples_validation % batch_size or batch_size])
    weights_val_sum = weights_val.sum()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    best_val_acc = -1
    best_val_epoch = -1
    print('Starting training.')
    for epoch in range(epochs):
        epoch_loss_val = 0
        epoch_acc_val = 0
        epoch_loss_train = 0
        epoch_acc_train = 0
        model.train()
        for i, (imgs, labels) in enumerate(cifar10_loader['train']):
            x, y = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(x)
            pred_class = (preds.argmax(dim=-1) == y).float()
            train_loss = loss_module(preds, y)
            train_loss.backward()
            optimizer.step()

            epoch_acc_train += weights_train[i] * pred_class.mean()
            epoch_loss_train += weights_train[i] * train_loss.detach().numpy()

        model.eval()
        for i, (imgs, labels) in enumerate(cifar10_loader['validation']):
            x, y = imgs.to(device), labels.to(device)
            with torch.no_grad():
                preds = model(x)
                pred_class = (preds.argmax(dim=-1) == y).float()
                val_loss = loss_module(preds, y)
                epoch_acc_val += weights_val[i] * pred_class.mean()
                epoch_loss_val += weights_val[i] * val_loss.detach().numpy()

        training_acc = epoch_acc_train/weights_train_sum
        training_loss = epoch_loss_train/weights_train_sum
        val_acc = epoch_acc_val/weights_val_sum
        val_loss = epoch_loss_val/weights_val_sum

        if val_acc > best_val_acc:
          best_model = deepcopy(model)
          best_val_epoch = epoch
          best_val_acc = val_acc

        logging_dict['loss_train'].append(training_loss)
        logging_dict['accuracy_train'].append(training_acc)
        logging_dict['loss_val'].append(val_loss)
        logging_dict['accuracy_val'].append(val_acc)

        print(f'Epoch: {epoch+1}, Training accuracy: {np.round(training_acc,2)}, Training loss: {training_loss:.4f}, Validation accuracy: {np.round(val_acc,2)}, Validation loss: {val_loss:.4f}')

    print('Training finished.')
    val_accuracies = logging_dict['accuracy_val']
    # TODO: Test best model
    test_metrics = evaluate_model(best_model if best_model is not None else model, cifar10_loader['test'])
    test_accuracy = test_metrics['accuracy']
    print(f'The final test accuracy of the best model at epoch {best_val_epoch} is {test_accuracy}.')
    # TODO: Add any information you might want to save for plotting
    logging_info = logging_dict
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    _, _, _, logging_info = train(**kwargs)
    loss_train = logging_info['loss_train']
    acc_train = logging_info['accuracy_train']
    loss_val = logging_info['loss_val']
    acc_val = logging_info['accuracy_val']

    plt.figure(figsize=(10, 5))

    epochs = range(1, len(loss_train) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_train, marker='o', label='Training Loss')
    plt.plot(epochs, loss_val, marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_train, marker='o', label='Training Accuracy')
    plt.plot(epochs, acc_val, marker='o', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # Feel free to add any additional functions, such as plotting of the loss curve here
    