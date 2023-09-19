#This code is an example of how pseudo-spot datasets were pre-processed in our project using TMM between-sample total gene counts adjustmens, log-transformation, ComBat batch correction
#and z-scaling. However this part was fit to our dataset and may not work well for other datasets. Thus, adapt pre-processing of your dataset.
#This code requires the pseudo-spot gene count and spatial matrix dataset from Pseudospotting_Tiling.py, the TMM factors from TMM_factors.py
#and the stain normalized H&E WSI from Stain_Normalization.py
#The first part of this code just runs pre-processing for a single sample, which means that it is to be repeated for the amount of samples you have.

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from anndata import AnnData
import anndata as ad
import scanpy as sc
import squidpy as sq

import os
vipshome = 'C:\\vips-dev-8.14\\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

from wsi_tile_cleanup import filters, utils
import glob


#Open TMM normalization factors
#------------------------------------------------------------------------------ 
with open('XXX.npy', 'rb') as f: #adjust this to the directory where the TMM factors were saved from the TMM_factors.py file
    norm = np.load(f)
#------------------------------------------------------------------------------ 


#Define loading data directory if needed
#------------------------------------------------------------------------------
load_data_dir = "XXX\\" #adjust this to your needs.
#------------------------------------------------------------------------------


#Get names of image tiles that are background in the H&E WSI
#------------------------------------------------------------------------------
Image.MAX_IMAGE_PIXELS = None
wsi_path = load_data_dir+"XXX.tif" #adjust to the directory where the stain normalized H&E WSI (from Stain_Normalization.py) was saved.
vi_wsi = utils.read_image(wsi_path, access="sequential")

otsu = filters.otsu_threshold(vi_wsi)

background_tiles = []
for filename in glob.glob(load_data_dir+'XXX\\*.tiff'): #adjust to the folder where the image tiles (from Pseudospotting_Tiling.py) are saved/located
    vi_bg = utils.read_image(filename)
    perc = filters.background_percent(vi_bg, otsu)
    if perc > 0.95: #adjust this percentage to your needs. Read the documentary of wsi_tile_cleanup if necessary for further explanation.
        background_tiles.append(filename)
#------------------------------------------------------------------------------


#Filter out pseudo-spots that are in background areas of the H&E WSI
#------------------------------------------------------------------------------
#Load raw counts gene matrix
gene_matrix = pd.read_csv(load_data_dir+"XXX.csv") #adjust this to the directory where the pseudo-spot gene count dataset (from Pseudospotting_Tiling.py) is saved.
gene_matrix.drop(columns="S100A6", inplace=True) #in our project S100A6 was dropped due to significantly larger gene expression than other genes. Can be adjusted to your needs.
gene_matrix['tile'] = gene_matrix['tile'].apply(lambda x: "{}{}".format(x, ".tiff"))

#Load spatial matrix
spatial_matrix = pd.read_csv(load_data_dir+"Raw_Spatial_Matrix_156_"+patient+".csv") #adjust this to the directory where the pseudo-spot spatial location dataset (from Pseudospotting_Tiling.py) is saved. 
spatial_matrix['tile'] = spatial_matrix['tile'].apply(lambda x: "{}{}".format(x, ".tiff"))

#Filter out background image tiles from the gene count and spatial matrix pseudo-spot dataset
for i in background_tiles:
    a = i.rsplit("\\")[-1]
    gene_matrix = gene_matrix[~gene_matrix.tile.str.contains(a)]
    spatial_matrix = spatial_matrix[~spatial_matrix.tile.str.contains(a)]

#Prepare gene and spatial matrix for anndata object
gene_matrix.set_index("tile", inplace=True)
spatial_matrix.set_index("tile", inplace=True)
spatial_array = spatial_matrix.to_numpy() #spatial matrix dataframe needs to be converted into a numpy array to be loaded into the AnnData object
#------------------------------------------------------------------------------


#Create AnnData object that includes the pseudo-spots gene counts and locations as well as the H&E WSI
#------------------------------------------------------------------------------
#Load H&E WSI
h_e_img = plt.imread(load_data_dir+"XXX.tif") #adjust to the directory where the stain normalized H&E WSI (from Stain_Normalization.py) was saved.

adata = AnnData(gene_matrix, obsm={"spatial": spatial_array})
spatial_key = "spatial"
library_id = "tissue"
adata.uns[spatial_key] = {library_id: {}}
adata.uns[spatial_key][library_id]["images"] = {}
adata.uns[spatial_key][library_id]["images"] = {"hires": h_e_img}
adata.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 130}
#------------------------------------------------------------------------------


#Calculate qc metrics and plot histplots of counts per pseudo-spot and n_genes per pseudo-spot
#------------------------------------------------------------------------------
sc.pp.calculate_qc_metrics(adata,percent_top=(50, 100, 150), inplace=True)

#Plot spatial plots of counts and n_genes per pseudo-spot
sq.pl.spatial_scatter(adata, shape="circle", color=["total_counts", "n_genes_by_counts"], 
                      cmap="bwr", alpha = 0.6, wspace=0.1,
                      scalebar_dx=0.72, scalebar_units="um", 
                      scalebar_kwargs={"fixed_value": 500, "location": "lower left", "box_alpha": 0.5})


#Plot histplots
fig, axs = plt.subplots(1, 2, figsize=(15, 4))
sns.histplot(
    adata.obs["total_counts"],
    kde=False,
    bins=100,
    ax=axs[0])
sns.histplot(
    adata.obs["n_genes_by_counts"],
    kde=False,
    bins=100,
    ax=axs[1])

#Plot histplots for the lower counts
fig, axs = plt.subplots(1, 2, figsize=(15, 4))
sns.histplot(
    adata.obs["total_counts"][adata.obs["total_counts"] < 50],
    kde=False,
    bins=50,
    ax=axs[0])
sns.histplot(
    adata.obs["n_genes_by_counts"][adata.obs["n_genes_by_counts"] < 50],
    kde=False,
    bins=50,
    ax=axs[1])
#------------------------------------------------------------------------------


#Filter low total counts pseudo-spot
#------------------------------------------------------------------------------
sc.pp.filter_cells(adata, min_counts=5) #min_counts is adjustable to your dataset and varies for each sample. Check histplots for the cutoff as well.
#Otherwise follow filtering steps that are suggested in Scanpy or Squidpy.

#Spatial plot of total_counts per pseudo-spot after filtering out low_counts pseudo-spots.
#Parameters of this needs to be adjusted depending on your dataset, resolution and so on. Read the documentary for this function on Squidpy.
sq.pl.spatial_scatter(adata, shape="circle", color=["total_counts"], 
                      cmap="bwr", alpha = 0.8, wspace=0.1, vmax=500,
                      scalebar_dx=0.72, scalebar_units="um", 
                      scalebar_kwargs={"fixed_value": 500, "location": "lower left", "box_alpha": 0.7})
#------------------------------------------------------------------------------


#Adjust spatial gene counts per pseudo-spot with sample-specific TMM factor and log-transformation
#------------------------------------------------------------------------------
adata.X = adata.X*norm[XXX] #adapt to the TMM factor for the specific sample that is processed.

#Log-transformation
sc.pp.log1p(adata)

#Save TMM and log-transformed AnnData object
adata.write(load_data_dir + "XXX") #adjust to your needs.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#The above code is an example for pre-processing each sample and only pre-processes one sample at a time. This part needs to be adapted to your dataset.
#The following code performs batch correction on all pre-processed samples. This means that you have to manually run the above code as many times as you have samples.
#All the saved TMM and log-transformed AnnData objects are necessary for the following code.
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#Load all TMM and log-transformed anndata objects
#------------------------------------------------------------------------------
YYY1 = ad.read_h5ad("XXX1.h5ad") #read the AnnData objects that were TMM and log-transformed before
YYY2 = ad.read_h5ad("XXX2.h5ad")
YYY3 = ad.read_h5ad("XXX3.h5ad")
#and so on
#------------------------------------------------------------------------------


#Batch effect Correction
#------------------------------------------------------------------------------
combined_adata = YYY1.concatenate(YYY2, YYY3, batch_key = "Patient") #adjust this code depending on the loaded AnnData objects that should be batch corrected

#Runs Combat batch effect correction. Read Scanpy documentation if necessary.
sc.pp.combat(combined_adata, key='Patient')
#------------------------------------------------------------------------------


#Scaling of gene counts per pseudo-spot
#------------------------------------------------------------------------------
#Use scaling if desired. Other data scaling methods can be used. In this case gene counts per pseudo-spot are scaled to 0 mean and unit variance.
scaler = StandardScaler() 
combined_adata.X = scaler.fit_transform(combined_adata.X)

#Save combined_adata if desired
combined_adata.write("XXX.h5ad") #adjust the directory to your needs.
#------------------------------------------------------------------------------


#Extract single patients and save the pre-processed (TMM, log-transform, Combat and z-scaling) pseudo-spot dataset
#------------------------------------------------------------------------------
single_adata = combined_adata[combined_adata.obs['Patient'].isin(['XXX']),:] #adjust to the sample that you want to extract from the combined AnnData object
 
single_adata.write("XXX.h5ad") #adjust the directory to your needs.
#------------------------------------------------------------------------------

#The outputs of the second part of this code are: Pre-processed combined AnnData objects (can be used for clustering and other downstream analyses) and single AnnData objects
#of each sample (are used for spatial gene expression prediction). The latter have the image tile name and all genes in columns and each row represents the pre-processed
#gene counts and location per pseudo-spot.
