#!/usr/bin/python3
#
#######################################################################
#                                                                     #
# INTRODUCTION                                                        #
#                                                                     #
#######################################################################
#
# This script downloads the "102 Category Flower Dataset" from the
# University of Oxford and saves the data in a directory structure
# which torchvision can manipulate. 
#
# A directory named "flowers" is created in the working directory.
# Data is organized with the following structure.
#
# flowers/
#    train/
#        1/         # Each directory here is the ID of a class.
#        2/         # (class IDs start at 1, not 0).
#        (...)
#        102/
#    test/
#        (...)
#    valid/
#        (...)
#
#######################################################################
#                                                                     #
# REFERENCES                                                          #
#                                                                     #
#######################################################################
#
# [1] University of Oxford 102 Flower Dataset
# https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
#
# [2] Torchivision ImageFolder class
# https://pytorch.org/vision/stable/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder
#
#######################################################################
#                                                                     #
# IMPORTS                                                             #
#                                                                     #
#######################################################################

import numpy as np
import os
from pathlib import Path
import re
from scipy.io import loadmat
import tarfile
from typing import List, Tuple
import wget

#######################################################################
#                                                                     #
# GLOBAL VARIABLES                                                    #
#                                                                     #
#######################################################################

# Dataset root directory
DATASET_ROOT = "flowers"

# File containing images
#
# Notes:
#
# 1 - This is a .tgz archive with the following format.
#
#            jpg/
#                image_00001.jpg
#                image_00002.jpg
#                (...)
#
#  2 - Image IDs start at 1, not 0.

DATASET_FILE_REMOTE = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
DATASET_FILE_LOCAL  = os.path.join(DATASET_ROOT, os.path.basename(DATASET_FILE_REMOTE))

# File containing image IDs for train/valid/test split.
#
# Notes:
#
# 1 - This is a Matlab .mat file.
# 2 - Image IDs start at 1, not 0.
# 3 - See get_train_ids() for procedure to retieve ids.

SPLIT_FILE_REMOTE = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
SPLIT_FILE_LOCAL  = os.path.join(DATASET_ROOT, os.path.basename(SPLIT_FILE_REMOTE))

# File containing class labels.
#
# Notes:
#
# 1 - Labels start at 1, not 0.
# 2 - See get_labels() for procedure to retrieve labels.

LABELS_FILE_REMOTE = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
LABELS_FILE_LOCAL  = os.path.join(DATASET_ROOT, os.path.basename(LABELS_FILE_REMOTE))

# Lists...

REMOTE_FILES = [DATASET_FILE_REMOTE, SPLIT_FILE_REMOTE, LABELS_FILE_REMOTE]
LOCAL_FILES  = [DATASET_FILE_LOCAL,  SPLIT_FILE_LOCAL,  LABELS_FILE_LOCAL]


#######################################################################
#                                                                     #
# FUNCTIONS                                                           #
#                                                                     #
#######################################################################


def create_root_directory() -> None:
    """Create directory. Print name of directory. Crash if exists."""

    message = "Creating directory '" + DATASET_ROOT + "'."
    print(message)
    Path(DATASET_ROOT).mkdir()


def download_remote_files() -> None:
    """Download list of files from remote url to local disk."""

    n = len(REMOTE_FILES)

    for i in range(n):

        remote_file = REMOTE_FILES[i]
        local_file = LOCAL_FILES[i]

        basename = os.path.basename(remote_file)

        print("Downloading file", basename, "...")
        wget.download(remote_file, out=local_file)
        print()


def delete_local_files() -> None:
    """Delete local versions of remote files."""

    for local_file in LOCAL_FILES:

        basename = os.path.basename(local_file)

        print("Removing file", basename, "...")
        os.remove(local_file)


def get_labels() -> np.array:
    """Return labels for each image."""

    # .mat file containing labels
    basename = os.path.basename(LABELS_FILE_LOCAL)
    path = os.path.join(DATASET_ROOT, basename)
    content = loadmat(path)
    
    labels = content["labels"][0,:]

    return labels 


def get_train_ids() -> Tuple[np.array, np.array, np.array]:
    """Return id images for train, valid, test. """
   
    # .mat file containing ids
    basename = os.path.basename(SPLIT_FILE_LOCAL)
    path = os.path.join(DATASET_ROOT, basename)
    content = loadmat(path)

    train_ids = content["trnid"][0,:]
    valid_ids = content["valid"][0,:]
    test_ids = content["tstid"][0,:]

    return train_ids, valid_ids, test_ids


def extract_files() -> None:
    """Extract files from archive to class subdirectories."""

    # Retrieve image class labels
    labels = get_labels()
    
    # Retieve image ids for train/valid/test split
    train_ids, valid_ids, test_ids = get_train_ids()

    # Create directories for each class
    # Unzip files to class directories
    
    print("Creating train/valid/test directories.")

    with tarfile.open(DATASET_FILE_LOCAL, "r") as t:

        members = t.getmembers()
        n = len(members)

        for i in range(n):

            member = members[i]

            # Ignore directories
            if member.isdir():
                continue

            # Extract image ID
            #
            # e.g. member.name -> jpg/image_0009.jpg
            #      image_name  -> image_0009.jpg
            #      image_id    -> 9
            
            image_name = os.path.basename(member.name)
            image_id = int(re.sub('[^0-9]',"", image_name))

            # Set member name as image name.
            # This eliminates the jpg/ directory prefix.
            # and will allow to extract images directly
            # into directories train/, valid/ and test/.
            member.name = image_name

            # Create target directory for this image
            target_dir = DATASET_ROOT

            # Append train/valid/test
            if image_id in train_ids:
                target_dir = os.path.join(target_dir, "train")
            elif image_id in valid_ids:
                target_dir = os.path.join(target_dir, "valid")
            elif image_id in test_ids:
                target_dir = os.path.join(target_dir, "test")
            else:
                raise Exception("unknown image id")

            # Append label
            label = labels[image_id-1]  # -1 since image ids start at 1
            target_dir = os.path.join(target_dir, str(label))

            # Create if does not exist.
            Path(target_dir).mkdir(parents=True, exist_ok=True)

            # Extract file into target directory.
            t.extract(member, target_dir) 


#######################################################################
#                                                                     #
# MAIN                                                                #
#                                                                     #
#######################################################################


def main():

    # Create dataset root directory (error if already exists).
    create_root_directory()

    # Download dataset files from the U of Oxford urls.
    download_remote_files()

    # Extract images from archive, splitting each image
    # into its corresponding train/valid/test directory.
    extract_files()

    # Delete local versions of remote files.
    delete_local_files()


if __name__ == "__main__":

    main()










