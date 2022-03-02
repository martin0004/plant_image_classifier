#!/usr/bin/python3
#
################################################################################
#                                                                              #
# IMPORTS                                                                      #
#                                                                              #
################################################################################

# Standard Libraries

import argparse
import os

# My Libraries

from utils import *

################################################################################
#                                                                              #
# GLOBAL VARIABLES                                                             #
#                                                                              #
################################################################################

# Parser arguments default values

DEFAULT_ARCH = "alexnet"
DEFAULT_SAVE_DIR = "."
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_HIDDEN_UNITS =  250
DEFAULT_EPOCHS = 30
DEFAULT_GPU = False

# Argument choices

SUPPORTED_ARCH = ["alexnet", "densenet161", "vgg16"]


################################################################################
#                                                                              #
# ARGUMENT PARSER                                                              #
#                                                                              #
################################################################################

def load_parser():
    
    description = "Train an image classifier from a directory of images."
    parser = argparse.ArgumentParser(description=description)
    
    # Positional arguments

    parser.add_argument("data_dir",
                        help="directory containing pictures for training")
    
    # Options
    
    parser.add_argument("--save_dir",
                        default=DEFAULT_SAVE_DIR, 
                        help="directory for saving trained models" \
                            +" (default: " + str(DEFAULT_SAVE_DIR) + ")")  
    
    parser.add_argument("--arch",
                        default=DEFAULT_ARCH,
                        choices=SUPPORTED_ARCH,
                        help="pre-trained model architecture" \
                            +" (default: " + str(DEFAULT_ARCH) + ")")  
    
    parser.add_argument("--learning_rate",
                        type=float,
                        default=DEFAULT_LEARNING_RATE,
                        help="training algorithm learning rate" \
                            +" (default: " + str(DEFAULT_LEARNING_RATE) + ")")   

    parser.add_argument("--hidden_units",
                        type=int, 
                        default=DEFAULT_HIDDEN_UNITS,
                        help="classifier hidden layers number of units" \
                            +" (default: " + str(DEFAULT_HIDDEN_UNITS) + ")")   
    
    parser.add_argument("--epochs",
                        type=int, 
                        default=DEFAULT_EPOCHS,
                        help="training algorithm number of epochs" \
                            +" (default: " + str(DEFAULT_EPOCHS) + ")")  
    
    # Flags
    
    parser.add_argument("--gpu",
                        action="store_true",
                        default=DEFAULT_GPU,
                        help="use GPU" \
                            +" (default: " + str(DEFAULT_GPU) + ")")   
    
    return parser


################################################################################
#                                                                              #
# MAIN                                                                         #
#                                                                              #
################################################################################

def main():
    
    # Parse command-line arguments
    
    parser = load_parser()
    args = parser.parse_args()
    
    # Output files
    
    metrics_file_path = os.path.join(args.save_dir, METRICS_FILE_NAME)
    checkpoint_file_path = os.path.join(args.save_dir, CHECKPOINT_FILE_NAME)
    
    # Load data loaders
    
    dls = get_data_loaders(args.data_dir)
    
    # Initialize untrained model
    
    n_categories = len(dls["train"].dataset.classes)
    model = get_model(args.arch, args.hidden_units, n_categories)
    
    # Train & validate model
    
    train_model(model,
                dls,
                args.learning_rate,
                args.hidden_units,
                args.epochs,
                args.gpu,
                metrics_file_path)

    # Save model checkpoint
    
    save_checkpoint(model,
                    dls["valid"].dataset.transform,
                    checkpoint_file_path,
                    args.arch,
                    args.hidden_units,
                    dls["train"].dataset.class_to_idx)
    
    # Plot learning curves

    plot_model_learning_curve(metrics_file_path)
    
    # Done!
    
    print("Training... done!")
    


if __name__ == "__main__":
    
    main()
