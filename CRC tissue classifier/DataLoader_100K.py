#This code file loads and transforms images from the NCT-CRC-HE-100K dataset so that they can be used for training and testing (different code files). Adjust the "XXX" for proper code usage. This files requires the download of the NCT-CRC-HE-100K dataset.

import random, os
import shutil
import zipfile
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix as conf_matrix
import random
from itertools import cycle

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional
from torch.utils.data import SubsetRandomSampler, ConcatDataset, DataLoader
import torchvision.transforms.functional as TF


#Set fixed seed
#------------------------------------------------------------------------------
DEFAULT_RANDOM_SEED = 42

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
    
seedEverything(seed=DEFAULT_RANDOM_SEED)
#------------------------------------------------------------------------------


#Set device to cuda or CPU
#------------------------------------------------------------------------------
device = ("cuda" if torch.cuda.is_available() else "cpu")
#------------------------------------------------------------------------------


#Set data directory and get date_time
#------------------------------------------------------------------------------
data_dir = 'XXX' #set directory where created files (e.g. training logs or figures) should be saved
date = str(datetime.today().strftime('%d%m%Y')) #set current date to add this information to saved files for better project tracking
#------------------------------------------------------------------------------


#Load Dataset
#------------------------------------------------------------------------------
#Basic transformation to be done on every imported image (train and testset)
basic_transform = transforms.Compose([transforms.Resize(224), #Resize to the expected input size of the used CNN model architectures
                                transforms.ToTensor(),
                                transforms.Normalize([0.7406, 0.5331, 0.7059], 
                                                     [0.1651, 0.2174, 0.1574])])
                                                     #Mean and standard deviation of the NCT-CRC-HE-100K dataset

#Load dataset with basic transformations
colondata = datasets.ImageFolder(
    "XXX", #insert the directory where the NCT-CRC-HE-100K dataset is stored
    transform=basic_transform)
#------------------------------------------------------------------------------


#Random train/validation/test splitting of fully loaded dataset
#------------------------------------------------------------------------------    
train_set, val_set, test_set = torch.utils.data.random_split(colondata, 
                                                             [0.7, 0.15, 0.15]) 
                                                             #adapt to desired splitting rations
#------------------------------------------------------------------------------  