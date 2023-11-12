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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import matplotlib.pyplot as plt
import cifar10_utils
import seaborn as sns

import torch


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
    n_classes = predictions.shape[1]
    conf_mat = np.zeros((n_classes, n_classes))

    pred = np.argmax(predictions, axis=1)

    for true_class in range(n_classes):
        for predicted_class in range(n_classes):
            conf_mat[true_class, predicted_class] = np.sum(np.logical_and(targets == true_class, pred == predicted_class))

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix of best model on the test dataset')
    plt.show()

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
    n_samples = np.sum(confusion_matrix)
    correct_pred = np.trace(confusion_matrix)
    accuracy = correct_pred / n_samples
    
    n_class = confusion_matrix.shape[0]
    precision = np.zeros(n_class)
    recall = np.zeros(n_class)
    f1_beta = np.zeros(n_class)

    for i in range(n_class):
      true_positives = confusion_matrix[i, i]
      false_positives = np.sum(confusion_matrix[:, i]) - true_positives
      false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

      precision[i] = true_positives / (true_positives + false_positives)
      recall[i] = true_positives / (true_positives + false_negatives)
      
      f1_beta[i] = (1 + beta**2) * (precision[i] * recall[i]) / (beta**2 * precision[i] + recall[i])
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_beta": f1_beta
    }

    print(f'The f1 score with beta {beta} for the best model on the test dataset is {f1_beta}.')
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
    data_prob = []
    data_y = []
    for batch in data_loader:
      x, y = batch
      prob = model.forward(x.reshape(x.shape[0], -1))
      data_y.extend(y.tolist())
      data_prob.extend(prob.tolist())
    
    conf_mat = confusion_matrix(np.array(data_prob), np.array(data_y))
    metrics = confusion_matrix_to_metrics(conf_mat)
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
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

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    print('Starting training.')
    logging_dict = {'loss_train': [], 'accuracy_train': [], 'loss_val': [], 'accuracy_val': []}
    best_val_acc = -1
    best_val_epoch = -1

    img_size = cifar10['train'][0][0].shape
    model = MLP(np.array(img_size).prod(), hidden_dims, 10)
    loss_module = CrossEntropyModule()

    n_samples_train = len(cifar10['train'])
    n_samples_validation = len(cifar10['validation'])
    num_batches_train = int(np.ceil(n_samples_train/batch_size))
    num_batches_val = int(np.ceil(n_samples_validation/batch_size))

    weights_train = np.array([batch_size] * (num_batches_train - 1) + [n_samples_train % batch_size or batch_size])
    weights_train_sum = weights_train.sum()
    weights_val = np.array([batch_size] * (num_batches_val - 1) + [n_samples_validation % batch_size or batch_size])
    weights_val_sum = weights_val.sum()

    for epoch in range(epochs):
      epoch_loss_val = 0
      epoch_acc_val = 0
      epoch_loss_train = 0
      epoch_acc_train = 0
      for i, sample in enumerate(cifar10_loader['train']):
        x, y = sample
        prob = model.forward(x.reshape(x.shape[0], -1))
        pred_class = np.argmax(prob, axis=1)
        epoch_acc_train += weights_train[i] * np.equal(pred_class, y).mean()
        epoch_loss_train += weights_train[i] * loss_module.forward(prob, y)
        dout = loss_module.backward(prob, y)
        model.backward(dout)
        for layer in model.layers:
          if hasattr(layer, 'params'):
            layer.params['weight'] -= lr*layer.grads['weight']
            layer.params['bias'] -= lr*layer.grads['bias']

        model.clear_cache()

      for i, sample in enumerate(cifar10_loader['validation']):
        x, y = sample
        prob = model.forward(x.reshape(x.shape[0], -1))
        pred_class = np.argmax(prob, axis=1)
        epoch_acc_val += weights_val[i] * np.equal(pred_class, y).mean()
        epoch_loss_val += weights_val[i] * loss_module.forward(prob, y)

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

      print(f'Epoch: {epoch+1}, Training accuracy: {training_acc:.2f}, Training loss: {training_loss:.4f}, Validation accuracy: {val_acc:.2f}, Validation loss: {val_loss:.4f}')
    
    print('Training is finished.')

    # TODO: Training loop including validation
    val_accuracies = logging_dict['accuracy_val']
    # TODO: Test best model
    metrics_test = evaluate_model(best_model if best_model is not None else model, cifar10_loader['test'])
    test_accuracy = metrics_test['accuracy']
    print(f'The final test accuracy of the best model at epoch {best_val_epoch} is {test_accuracy}.')
    # TODO: Add any information you might want to save for plotting
    logging_dict['metrics_test'] = metrics_test
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

    _, _, test_acc, logging_info = train(**kwargs)
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
    