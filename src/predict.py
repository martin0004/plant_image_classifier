#!/usr/bin/python3
#
################################################################################
#                                                                              #
# IMPORTS                                                                      #
#                                                                              #
################################################################################

# Standard Libraries

import argparse

# Third-Party Librairies

import torch

# My Libraries

from utils import *

################################################################################
#                                                                              #
# GLOBAL VARIABLES                                                             #
#                                                                              #
################################################################################

# Parser arguments default values

DEFAULT_TOP_K = 5
DEFAULT_GPU = False


################################################################################
#                                                                              #
# ARGUMENT PARSER                                                              #
#                                                                              #
################################################################################

def load_parser():
    
    description = "Predict an image class from a pre-trained model."
    parser = argparse.ArgumentParser(description=description)
    
    # Positional arguments

    parser.add_argument("image_path",
                        help="path to image file")
    
    parser.add_argument("checkpoint_path",
                        help="path to trained model checkpoint file")
    
    # Options
    
    parser.add_argument("--top_k",
                        type=int,
                        default=DEFAULT_TOP_K, 
                        help="return top k most likely classes" \
                            +" (default: " + str(DEFAULT_TOP_K) + ")")  

    parser.add_argument("--category_names",
                        help="path to JSON file mapping category integers to names")     
    
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
    
    # Dictionary of categories
    
    d_categories = get_categories(args.category_names)
    
    # Load checkpoint
    
    model, tf = load_checkpoint(args.checkpoint_path)
    
    # Predict classes
    
    probs_top_k, category_ids_top_k = predict_top_k_categories(args.image_path,
                                                               model,
                                                               tf,
                                                               args.top_k,
                                                               args.gpu)
    
    # Plot predictions
    
    plot_top_k_categories(args.image_path, d_categories, probs_top_k, category_ids_top_k)
    
    # Done!
    
    print("Predicting... done!")
    

if __name__ == "__main__":
    
    main()
    
    
    