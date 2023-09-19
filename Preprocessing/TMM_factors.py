#This file calculates trimmed mean of M values normalization factors for each sample that can be used to adjust spatial transcriptomics gene counts based on the differences of total gene counts across all samples. 
#This requires samples' / patients' .csv files with at least the columns: "x", "y" and "gene", where "x" and "y" stand for the 2D-location of a specific "gene" from the spatial transcriptomics dataset 
#matching the coordinates of the H&E whole-slide image. Each row stands for one gene at a specific location. Note that this code works for in situ sequencing datasets and would need major adaptions if
#spatial barcoding (e.g. 10x Visium) datasets are present. Adjust "XXX" to your needs.

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from anndata import AnnData
import scanpy as sc
import squidpy as sq
from rnanorm import TMM

#Get TMM normalization factors
#------------------------------------------------------------------------------ 
patients = ["p007", "p008", "p009", "p013", "p014", "p016", "p020", "p021", "p025", "p026"] #these were the existing samples / patients in the present project. Adjust to your needs.

#Create empty dataframes for bulked gene counts per sample that are filled later. Adjust number and names to the amount of existing samples / patients in your project
bulked_p007 = pd.DataFrame()
bulked_p008 = pd.DataFrame()
bulked_p009 = pd.DataFrame()
bulked_p013 = pd.DataFrame()
bulked_p014 = pd.DataFrame()
bulked_p016 = pd.DataFrame()
bulked_p020 = pd.DataFrame()
bulked_p021 = pd.DataFrame()
bulked_p025 = pd.DataFrame()
bulked_p026 = pd.DataFrame()
bulked_patients = [bulked_p007, bulked_p008, bulked_p009, bulked_p013, bulked_p014, bulked_p016, bulked_p020, bulked_p021, bulked_p025, bulked_p026]

#Iterate through the different samples and load the sum of all counts per gene into the corresponding bulked_pXXX dataframe
i=0
for patient in patients:
    load_data_dir = "XXX" #adjust to your directory
    genes = pd.read_csv(load_data_dir + "XXX.csv") #load your gene_location .csv file (as stated in the header)
    bulked_patients[i] = pd.DataFrame(genes["gene"].value_counts()) #sums up all counts per gene in the .csv file and adds the sum to the corresponding bulked dataframe
    bulked_patients[i].sort_index(ascending=True, inplace=True)
    bulked_patients[i] = bulked_patients[i].rename(columns={'gene': 'genes_' + patient})
    i+=1
    
all_patients = pd.concat(bulked_patients, axis=1) #concat all bulked dataframes from all samples

all_patients.drop("S100A6", axis="index", inplace=True) #in the present project S100A6 was removed due to significantly larger expression. Adjustable to your needs.

#Calculate TMM factors for each sample / patient
all_patients_arr = all_patients.to_numpy()
all_patients_arr = np.transpose(all_patients_arr)
x = TMM().fit(all_patients_arr)
norm_factors = x.get_norm_factors(all_patients_arr)

#Saving TMM factors
with open('XXX\\TMM_factors.npy', 'wb') as f:
    np.save(f, norm_factors)
#------------------------------------------------------------------------------ 

#The output of this codefile is: saved TMM factors (equalling the number of existing samples / patients) as .npy file
