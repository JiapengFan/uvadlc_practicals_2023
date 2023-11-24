################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from torch.utils.data import DataLoader
import os
import json

from cifar100_utils import get_train_validation_set, get_test_set

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    # Randomly initialize and modify the model's last layer for CIFAR100.
    num_in = model.fc.in_features
    model.fc = nn.Linear(num_in, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True
    torch.nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    torch.nn.init.zeros_(model.fc.bias)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model

def get_dataloader_train_val(dataset, batch_size):
    train_dataloader      = DataLoader(dataset=dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(dataset=dataset["validation"], batch_size=batch_size, shuffle=False, drop_last=False)
    return {"train": train_dataloader, "validation": validation_dataloader}

def get_dataloader_test(dataset, batch_size):
    test_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return test_dataloader

def _get_model_file(model_path, model_name):
    # Name of the file for storing network parameters
    return os.path.join(model_path, model_name + ".tar")

def load_model(model, model_path, model_name):
    """
    Loads a saved model from disk.
    """
    model_file = _get_model_file(model_path, model_name)
    assert os.path.isfile(model_file), f"Could not find the model file \"{model_file}\". Are you sure this is the correct path and you have your model stored here?"
    model.load_state_dict(torch.load(model_file))
    return model

def save_model(model, model_path, model_name):
    """
    Given a model, we save the state_dict and hyperparameters.
    """
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, model_name + '.tar')
    torch.save(model.state_dict(), model_file)

def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    model_path = 'saved_models'

    # Load the datasets
    cifar10_train, cifar_val = get_train_validation_set(data_dir, augmentation_name=augmentation_name)
    cifar10 = {'train': np.array(cifar10_train), 'validation': np.array(cifar_val)}
    cifar10_dataloader = get_dataloader_train_val(cifar10, batch_size)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_module = nn.CrossEntropyLoss()
    
    # Training loop with validation after each epoch. Save the best model.
    n_samples_train = len(cifar10['train'])
    n_samples_validation = len(cifar10['validation'])
    num_batches_train = int(np.floor(n_samples_train/batch_size))   # Drop last is true
    num_batches_val = int(np.ceil(n_samples_validation/batch_size))

    weights_train = np.array([batch_size] * (num_batches_train - 1) + [batch_size % batch_size or batch_size])
    weights_train_sum = weights_train[:-1].sum()    # Drop last is true
    weights_val = np.array([batch_size] * (num_batches_val - 1) + [n_samples_validation % batch_size or batch_size])
    weights_val_sum = weights_val.sum()     # Drop last is false
    best_val_acc = -1
    best_val_epoch = -1
    print('Starting training.')
    for epoch in range(epochs):
        epoch_loss_val = 0
        epoch_acc_val = 0
        epoch_loss_train = 0
        epoch_acc_train = 0
        model.train()
        for i, (imgs, labels) in enumerate(cifar10_dataloader['train']):
            x, y = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(x)
            bool_pred = (preds.argmax(dim=-1) == y).float()
            train_loss = loss_module(preds, y)
            train_loss.backward()
            optimizer.step()

            epoch_acc_train += weights_train[i] * bool_pred.mean()
            epoch_loss_train += weights_train[i] * train_loss.detach().cpu().numpy()

        model.eval()
        for i, (imgs, labels) in enumerate(cifar10_dataloader['validation']):
            x, y = imgs.to(device), labels.to(device)
            with torch.no_grad():
                preds = model(x)
                pred_class = (preds.argmax(dim=-1) == y).float()
                val_loss = loss_module(preds, y)
                epoch_acc_val += weights_val[i] * pred_class.mean().detach().cpu().numpy()
                epoch_loss_val += weights_val[i] * val_loss.detach().cpu().numpy()

        training_acc = epoch_acc_train/weights_train_sum
        training_loss = epoch_loss_train/weights_train_sum
        val_acc = epoch_acc_val/weights_val_sum
        val_loss = epoch_loss_val/weights_val_sum

        if val_acc > best_val_acc:
          save_model(model, model_path, checkpoint_name)
          best_val_epoch = epoch
          best_val_acc = val_acc

        print(f'Epoch: {epoch+1}, Training accuracy: {np.round(training_acc,2)}, Training loss: {training_loss:.4f}, Validation accuracy: {np.round(val_acc,2)}, Validation loss: {val_loss:.4f}')
    print('Training finished.')

    # Load the best model on val accuracy and return it.
    print(f'Loading the best performing model on the validation data at Epoch {best_val_epoch} with validation accuracy {val_acc}.')
    model = load_model(model, model_path, checkpoint_name)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_pred = []
    data_y = []
    for imgs, labels in data_loader:
      x, y = imgs.to(device), labels.to(device)
      with torch.no_grad():
        preds = model(x)
        pred_class = (preds.argmax(dim=-1) == y).float()
        data_y.extend(y.tolist())
        data_pred.extend(pred_class.tolist())

    accuracy = np.mean(np.array(data_y) == np.array(data_pred))

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = get_model()
    model.to(device)
    model_name = model.__class__.__name__

    # Get the augmentation to use
    cifar10_test = get_test_set(data_dir, test_noise)
    cifar10_test_dataloader = get_dataloader_test(cifar10_test, batch_size)

    # Train the model
    model = train_model(model, lr, batch_size, epochs, data_dir, model_name, device, augmentation_name)

    # Evaluate the model on the test set
    acc = evaluate_model(model, cifar10_test_dataloader, device)
    print(f'The test accuracy of the best performing model is {acc}.')

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
