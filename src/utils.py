#!/usr/bin/python3
#
################################################################################
#                                                                              #
# IMPORTS                                                                      #
#                                                                              #
################################################################################

# Standard Libraries

import csv
import json
import os
from pprint import pprint
import sys  
from typing import Dict, List, Tuple

# Third Party Libraries

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as psub
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import *
from torchvision.transforms import *


################################################################################
#                                                                              #
# GLOBAL VARIABLES                                                             #
#                                                                              #
################################################################################

METRICS_FILE_NAME = "metrics.csv"       # Metrics file name
CHECKPOINT_FILE_NAME = "checkpoint.pth" # Model checkpoint file name


################################################################################
#                                                                              #
# CATEGORIES                                                                   #
#                                                                              #
################################################################################

def get_categories(file_path: str) -> Dict:
    """Load categories id/name pairs from JSON file."""
    
    with open(file_path, "r") as f:
        d_categories = json.load(f)
        
    return d_categories  


def get_image_category_id_and_name(d_categories: Dict,
                                   dt: ImageFolder,
                                   image_index: int) -> Tuple[str, str]:
    """Find category id/name of an image from an image folder."""
    
    category_id = None
    category_name = None
    
    category_index = dt[image_index][1]        # Integer representing order of class in list - start at 0
    category_id = dt.classes[category_index]   # Integer stored as string associated to name - start at 1
    category_name = d_categories[category_id]  # Name of category (i.e. word representing plant species)
    
    return category_id, category_name


################################################################################
#                                                                              #
# DATALOADERS                                                                  #
#                                                                              #
################################################################################

def get_data_loaders(data_dir: str) -> Dict:
    """Build dataloaders from a directory of images.."""

    # Parameters

    means = [0.485, 0.456, 0.406]      # Mean value for normalizing RGB image (ImageNet values)
    stds = [0.229, 0.224, 0.225]       # Standard deviations for normalizing RGB image (ImageNet values)

    px_resize = 255                    # Length of shortest image side after resize
    px_crop = 224                      # Width & height for crop
    
    # Initialize dataloaders
    
    dls = dict()
    
    dls["train"] = None
    dls["valid"] = None
    dls["test"] = None

    # Training dataloader

    directory = data_dir + "/train"   
    
    tf = Compose([RandomRotation(45),
                  RandomResizedCrop(px_crop),
                  RandomHorizontalFlip(),
                  ToTensor(),
                  Normalize(means,stds)])

    dt = ImageFolder(directory, transform=tf)
    dl = DataLoader(dt, batch_size=32, shuffle=True)
    dls["train"] = dl
    
    # Validation dataloader
    
    directory = data_dir + "/valid"
    
    tf = Compose([Resize(px_resize),
                  CenterCrop(px_crop),
                  ToTensor(),
                  Normalize(means,stds)])
    
    dt = ImageFolder(directory, transform=tf)
    dl = DataLoader(dt, batch_size=32)
    dls["valid"] = dl
    
    # Testing dataloader
        
    directory = data_dir + '/test'
    dt = ImageFolder(directory, transform=tf) # Same transforms as validation
    dl = DataLoader(dt, batch_size=32)
    dls["test"] = dl
    
    return dls


################################################################################
#                                                                              #
# MODEL                                                                        #
#                                                                              #
################################################################################


def get_model(arch: str,
              n_hidden_units: int,
              n_classes: int):
    """Load a torchvision pre-trained model with a new untrained
    classifier.
    """
    
    # Check if model is in list of supported models
    
    supported_arch = ["alexnet", "densenet161", "vgg16"]
    
    if arch not in supported_arch:
        
        message = "Model " + arch + " not supported by this application."
        raise ValueError(message)
    
    
    # Model training

    p = 0.50          # Dropout probability    
    
    # Load torchvision pre-trained model.
    #
    # Find available values for variable arch here:
    # https://pytorch.org/vision/stable/models.html
    #
    # e.g.   Load alexnet model with
    #        get_torchvision_model("alexnet")
    
    line = arch + "(pretrained='True')"
    model = eval(line)   
    
    # Find number of input features of model classifier.

    if arch == "alexnet":
        n_features = model.classifier[1].in_features    
    elif arch == "densenet161":
        n_features = model.classifier.in_features
    elif arch == "vgg16":  
        n_features = model.classifier[0].in_features
    
    # Create classifier.
    
    classifier = nn.Sequential(nn.Linear(n_features, n_hidden_units, bias=True),
                               nn.ReLU(),
                               nn.Dropout(p),
                               nn.Linear(n_hidden_units, n_hidden_units, bias=True),
                               nn.ReLU(),
                               nn.Dropout(p),
                               nn.Linear(n_hidden_units, n_classes, bias=True),
                               nn.LogSoftmax(dim=1))
    
    # Swap pre-trained model classifier with new classifier.

    model.classifier = classifier
    
    return model


################################################################################
#                                                                              #
# METRICS                                                                      #
#                                                                              #
################################################################################

def print_epoch_metrics(learning_rate, n_hidden_units, dataset, epoch, acc, loss):
    """Print current epoch metrics to screen."""
    
    print("learning_rate: ", format(learning_rate, ".3f"),
          "n_hidden_units: ", n_hidden_units,
          "dataset: ", dataset,
          "epoch: ", epoch,
          "accuracy: ", format(acc, ".3f"),
          "loss: ", format(loss, ".3f"))    

    
def save_epoch_metrics(learning_rate, hidden_units, dataset, epoch, acc, loss, file_metrics_path):
    """Save current epoch metrics to file."""

    # Check if CSV file exist. If not, initialize.
    
    if not os.path.exists(file_metrics_path):   
            
        with open(file_metrics_path, "w") as f:
            
            headers = ["learning_rate", "hidden_units", "dataset", "epoch", "acc", "loss"]
            
            writer = csv.writer(f)            
            writer.writerow(headers)     
    
    # Append metrics to CSV file.
    
    with open(file_metrics_path, "a") as f:
        
        data = [learning_rate, hidden_units, dataset, epoch, acc, loss]
        
        writer = csv.writer(f)
        writer.writerow(data)    


def load_metrics(file_metrics_path: str) -> pd.DataFrame:
    """Load metrics file content into DataFrame"""
    
    df_metrics = pd.read_csv(file_metrics_path)
    
    return df_metrics
    
    
################################################################################
#                                                                              #
# TRAINING                                                                     #
#                                                                              #
################################################################################     


def get_accuracy_increment(log_p: float,
                           labels: np.array,
                           n_data: int) -> float:
    """Derive the accuracy increase at each step of model training."""
    
    p = F.softmax(log_p, dim=1)
    p_max, index_max = p.topk(1, dim=1)
    index_max = index_max.view(-1)
    correct_labels = (index_max == labels)                    # Tensor of bools
    correct_labels = correct_labels.type(torch.FloatTensor)   # Tensor of floats
    
    acc_increment = correct_labels.sum().item() / n_data
    
    return acc_increment    


def train_model(model,
                dls: Dict,
                learning_rate: float,
                n_hidden_units: int,
                n_epochs: int,
                gpu: bool,
                file_metrics_path: str):
    """Train and validate a torchvision neural network."""
    
    # Select device (CPU or GPU)
    
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)

    # Define optimizer
    
    optimizer = torch.optim.SGD(model.classifier.parameters(), learning_rate)
    
    # Select loss function
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    # Some constants...
    
    n_train = len(dls["train"].dataset)            # Number of data points in training dataset
    n_valid = len(dls["valid"].dataset)            # Number of data points in validation dataset
    
    # Train
    
    for e in range(n_epochs):

        model.train()         # Enable dropout
        
        batch_mean_loss = 0   # Loss (mean) - entire batch - current epoch    
        acc = 0               # Accuracy - entire batch - current epoch
        
        for images, labels in dls["train"]:
            
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()    
            
            # Loss
        
            log_p = model(images)   
        
            mini_batch_total_loss = loss_function(log_p, labels)      
            mini_batch_total_loss.backward()
            batch_mean_loss += mini_batch_total_loss.item() / n_train
        
            optimizer.step()  
            
            # Accuracy
                                
            acc += get_accuracy_increment(log_p, labels, n_train)
        
        # Print and save metrics        
        
        dataset = "train"        
        print_epoch_metrics(learning_rate, n_hidden_units, dataset, e, acc, batch_mean_loss)
        save_epoch_metrics(learning_rate, n_hidden_units, dataset, e, acc, batch_mean_loss, file_metrics_path)
        
        # Validate
        
        with torch.no_grad():     # Disable gradient calculation
            
            model.eval()          # Disable dropout
            
            batch_mean_loss = 0   # Loss (mean) - entire batch - current epoch    
            acc = 0               # Accuracy - entire batch - current epoch
        
            for images, labels in dls["valid"]:
                
                images = images.to(device)
                labels = labels.to(device)

                # Loss
        
                log_p = model(images)   
        
                mini_batch_total_loss = loss_function(log_p, labels)     
                batch_mean_loss += mini_batch_total_loss.item() / n_valid

                # Accuracy
                
                acc += get_accuracy_increment(log_p, labels, n_valid)
        
            # Print and save metrics        
        
            dataset = "valid"        
            print_epoch_metrics(learning_rate, n_hidden_units, dataset, e, acc, batch_mean_loss)
            save_epoch_metrics(learning_rate, n_hidden_units, dataset, e, acc, batch_mean_loss, file_metrics_path)


def plot_model_learning_curve(file_metrics_path: str) -> None:
    """Plot model learning curve from metrics file."""
    
    # Load metrics
    
    df = load_metrics(file_metrics_path)
    
    # Metrics for train / valid datasets
    
    df_train = df[df["dataset"] == "train"]
    df_valid = df[df["dataset"] == "valid"]
    
    # Figure parameters
    
    n_rows = 1
    n_cols = 2
    
    # Subplot titles
    
    subplot_titles = ["accuracy", "loss"]   
    
    # Create figure
    
    fig = psub.make_subplots(rows=n_rows, cols=n_cols, subplot_titles = subplot_titles)
    
    # Plot accuracy
    
    df = df_train
    trace = go.Scatter(x=df["epoch"], y=df["acc"], mode="lines", name="train", line_color = "green")
    fig.add_trace(trace, row=1, col=1)  
    
    df = df_valid
    trace = go.Scatter(x=df["epoch"], y=df["acc"], mode="lines", name="valid", line_color = "blue")
    fig.add_trace(trace, row=1, col=1)      
    
    fig.layout.xaxis1.title = "epoch"
    fig.layout.yaxis1.title = "accuracy"    
    
    # Plot loss
    
    df = df_train
    trace = go.Scatter(x=df["epoch"], y=df["loss"], mode="lines", name="train", line_color = "green")
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)  
    fig.layout.xaxis2.title = "epoch"
    
    df = df_valid
    trace = go.Scatter(x=df["epoch"], y=df["loss"], mode="lines", name="valid", line_color = "blue")
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)  
    
    fig.layout.xaxis2.title = "epoch"
    fig.layout.yaxis2.title = "loss"       
    
    fig.show()            
            
            
################################################################################
#                                                                              #
# CHECKPOINTS                                                                  #
#                                                                              #
################################################################################

    
def save_checkpoint(model,
                    tf, 
                    file_checkpoint_path: str,
                    arch: str,
                    n_hidden_units: int,
                    class_to_idx: Dict) -> None:
        
    # Move model to CPU
    #
    # (I've had some issues in this project when saving
    #  a model in GPU mode then trying to load it in
    #  CPU mode. So I'm saving to CPU and only using
    #  GPU mode for calculations.)
    
    model.to(torch.device("cpu"))

    # Define checkpoint
    
    d_checkpoint = dict()
    
    d_checkpoint["state_dict"] = model.state_dict()
    d_checkpoint["arch"] = arch
    d_checkpoint["hidden_units"] = n_hidden_units
    d_checkpoint["n_classes"] = len(class_to_idx)
    d_checkpoint["class_to_idx"] = class_to_idx
    d_checkpoint["transforms"] = tf
    
    # Save checkpoint

    torch.save(d_checkpoint, file_checkpoint_path)            


def load_checkpoint(file_checkpoint_path: str):
    
    # Load checkpoint
    
    d_checkpoint = torch.load(file_checkpoint_path)    
    
    # Retrieve model
    
    model = get_model(d_checkpoint["arch"], 
                      d_checkpoint["hidden_units"], 
                      d_checkpoint["n_classes"])
    
    model.load_state_dict(d_checkpoint["state_dict"])
    
    model.class_to_idx = d_checkpoint["class_to_idx"]
    
    # Retrieve transforms
    
    tf = d_checkpoint["transforms"]
    
    # Return model
    
    return model, tf


################################################################################
#                                                                              #
# PREDICTIONS                                                                  #
#                                                                              #
################################################################################


def predict_top_k_categories(file_input_path: str,
                             model,
                             tf,
                             top_k: int,
                             gpu: bool):
    ''' Predict the most likely classes of an image using a neural network.
    '''
    
    # Load image

    with Image.open(file_input_path) as im:
        
        im_torch = tf(im)
        im_torch = im_torch.view(1, *im_torch.shape)  # Reshape to format required by model
    
    # Select device
    
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    im_torch = im_torch.to(device)    
         
    with torch.no_grad():     # Disable gradient calculation

        model.eval()          # Disable dropout  
        log_p = model(im_torch)    
  
    p = torch.exp(log_p)
    
    probs, category_indexes = p.topk(top_k)
    
    probs = probs.view(-1).tolist()
    category_indexes = category_indexes.view(-1).tolist()
    
    category_ids = []
    for index in category_indexes:
        for key, value in model.class_to_idx.items():
            if value == index:
                category_ids.append(key)
    
    return probs, category_ids    


def plot_top_k_categories(file_input_path: str,
                          d_categories: Dict,
                          probs_top_k: List,
                          category_ids_top_k: List) -> None:
    """Print bar chart of most probable categories of an image."""
    
    # Figure properties
    
    n_rows = 1
    n_cols = 2
    
    figure_height = 400
    figure_width = n_cols * figure_height
    
    # Subplot titles
    
    subplot_titles = ["Original Image", "Predicted Species"]
    
    # Create figure
    
    fig = psub.make_subplots(rows=n_rows, cols=n_cols, subplot_titles = subplot_titles, horizontal_spacing=0.2)
    
    # Create subplots
    
    im = Image.open(file_input_path)
    trace = go.Image(z=im)
    fig.add_trace(trace, row=1, col=1)
    fig.layout.xaxis1.showticklabels = False
    fig.layout.yaxis1.showticklabels = False
    
    category_names_top_k = [ d_categories[id] for id in category_ids_top_k]
    trace = go.Bar(x=probs_top_k, y = category_names_top_k, orientation="h", marker_color="blue")
    fig.add_trace(trace, row=1, col=2)
    fig.layout.yaxis2.title = "probability"
    
    # Show figure
    
    fig.layout.width = figure_width
    fig.layout.height = figure_height
    
    fig.show()

