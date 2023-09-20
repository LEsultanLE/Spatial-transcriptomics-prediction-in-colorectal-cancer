#This code file loads and transforms image tiles as well as pre-processed gene counts of a single gene of interest from the 
#pre-processed AnnData objects from Preprocessing_Tiling.py for validation. Here, no data augmentations are used. Adjust the "XXX" for proper code usage.
#This code requires the pre-processed AnnData objects and all image tiles from each sample. This only runs for one sample, so you have to re-run it for
#every other sample.

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
from sklearn.model_selection import train_test_split
import random
from itertools import cycle
from scipy import stats
from sklearn.metrics import r2_score

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional
from torch.utils.data import SubsetRandomSampler, ConcatDataset, DataLoader
import torchvision.transforms.functional as TF

from anndata import AnnData
import anndata as ad
import scanpy as sc
import squidpy as sq


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


#Get date
#------------------------------------------------------------------------------
date = str(datetime.today().strftime('%d%m%Y'))
#------------------------------------------------------------------------------


#Load pre-processed AnnData object(s) and filter out the gene that you want to predict
#------------------------------------------------------------------------------
#Define the columns of the dataframe that is later used to extract the input data for the models
columns_of_interest = ["tile", "RAMP1"]

sample = "p013" #This was the name of one of the samples used in our project as validation samples. Adjust to your needs.

#generate training dataframe with all training samples
adata = ad.read_h5ad("XXX/"+sample+".h5ad") #adjust directory to your needs where transformed AnnData objects (from Preprocessing_Example.py) are saved
st_dataset = adata.to_df()
st_dataset["tile"] = st_dataset.index
  
st_dataset_filtered = st_dataset.copy()
st_dataset_filtered = st_dataset_filtered[columns_of_interest] #filter for columns of interest (i.e. tile and genes of interest)

if sample == "p009":
  st_dataset_filtered['tile'] = st_dataset_filtered['tile'].str.replace('-2', '')
	#in our case random strings were added to the name of each image tile. These needed to be deleted so that the names in the st_dataset exactly match the names of the saved image tiles.
elif sample == "p013":
  st_dataset_filtered['tile'] = st_dataset_filtered['tile'].str.replace('-3', '')

st_dataset_filtered['tile'] = st_dataset_filtered['tile'].apply(lambda x: "{}{}".format("XXX"))
#adjust XXX in such a way that the name of the image tile in st_dataset_filtered exactly matches the name (including directory) of the H&E image tiles. 
#Later the models will load the image tiles by the name they find in the st_dataset.

#reset index of dataframes
st_dataset_filtered.reset_index(drop=True, inplace=True)

#get genes of interest
gene_list = list(st_dataset_filtered)[1:]
#------------------------------------------------------------------------------


#Load images with basic transforms
#------------------------------------------------------------------------------
class STDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        gene_names = list(self.dataframe)[1:]
        gene_vals = []
        row = self.dataframe.iloc[index]
        transform = transforms.Compose([
              transforms.Resize((224, 224)), 
              transforms.ToTensor(), 
              transforms.Normalize([0.7406, 0.5331, 0.7059], [0.1651, 0.2174, 0.1574]) #apply normalization transforms as for CRC tissue classifier
              ])
        a= Image.open(row["tile"]).convert("RGB")
        #print(x.size)
        for j in gene_names:
            gene_val = float(row[j])
            gene_vals.append(gene_val)
        e=row["tile"]
        a = transform(a)
        return (a, gene_vals, e)
    

loaded_valid_dataset = STDataset(st_dataset_filtered)
#------------------------------------------------------------------------------