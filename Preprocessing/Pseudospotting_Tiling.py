#This code tiles the stain normalized H&E WSI and creates a pseudo-spot spatial transcriptomics dataset from the in situ sequencing dataset where each pseudo-spot matches the corresponding image tile from the H&E WSI.
#This requires samples' / patients' .csv files with at least the columns: "x", "y" and "gene", where "x" and "y" stand for the 2D-location of a specific "gene" from the spatial transcriptomics dataset 
#matching the coordinates of the H&E whole-slide image. Each row stands for one gene at a specific location. Note that this code works for in situ sequencing datasets and would need major adaptions if
#spatial barcoding (e.g. 10x Visium) datasets are present. It additionally needs the stain normalized H&E WSI.

from PIL import Image

import random, os
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional

#This step is necessary when trying to import openslide package on Windows. Adjust XXX to the directory where the package is installed.
OPENSLIDE_PATH = "XXX\\Lib\\site-packages\\openslide-win64-20230414\\bin" 
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator


#Define loading data directory
#------------------------------------------------------------------------------
load_data_dir = "XXX\\" #adjust this to your needs.
#------------------------------------------------------------------------------


#Load stain normalized H&E WSI and tile it (as tiles need to be fed into the models later).
#------------------------------------------------------------------------------
#Use this line of code so that very large images can be loaded without warning message.
Image.MAX_IMAGE_PIXELS = None
img_slide = open_slide(load_data_dir + "XXX.tif") #adjust this to the stain normalized H&E WSI

tiles = DeepZoomGenerator(img_slide, tile_size=156, overlap=0, limit_bounds=False) 
#adjust tile_size depending on the resolution (i.e. um/pixel) of your dataset. In our dataset the tile_size of 156 matched the tile_size of the NCT-CRC-HE-100K dataset.

#Get level_count of the generated tiles
tiles.level_count
cols, rows = tiles.level_tiles[14] #select the highest number in [XX] to get the smallest image tiles. Check openslide documentary if necessary.
#------------------------------------------------------------------------------


#Load Gene data
#This requires samples' / patients' .csv files with at least the columns: "x", "y" and "gene", where "x" and "y" stand for the 2D-location of a specific "gene" from the spatial transcriptomics dataset 
#matching the coordinates of the H&E whole-slide image. Each row stands for one gene at a specific location. Note that this code works for in situ sequencing datasets and would need major adaptions if
#spatial barcoding (e.g. 10x Visium) datasets are present.
#------------------------------------------------------------------------------
gene_coords = pd.read_csv(load_data_dir+"XXX.csv") #adjust this to your needs.
#------------------------------------------------------------------------------


#Create empty dataframe for pseudo-spot dataset for gene counts per pseudo-spot and spatial location of each pseudo-spot.
#------------------------------------------------------------------------------
img_tile_save_dir = load_data_dir + "XXX\\" #adjust to where image tiles should be saved
gene_tile_df = pd.DataFrame(columns=["tile"])
gene_tile_df = gene_tile_df.astype("int64")

#creates an empty spatial matrix dataframe that has the center location of each pseudo-spot in a row
spatial_tile_df = pd.DataFrame(columns=["tile","x", "y"])
spatial_tile_df = spatial_adata.astype("int64")
#------------------------------------------------------------------------------


#Tile H&E WSI and sum up all gene counts from the in situ sequencing dataset into a pseudo-spot that matches each specific image tile.
#------------------------------------------------------------------------------
for row in range(rows):
    for col in range(cols):
        tile_name = os.path.join(img_tile_save_dir, '%d_%d' % (col, row))
        temp_tile = tiles.get_tile(14, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        plt.imsave(tile_name + ".tiff", temp_tile_np) #saves each image tile
        
        gene_subs = gene_coords[(gene_coords["x"]>=col*156) & 
                                (gene_coords["x"] < (col+1)*156) & 
                                (gene_coords["y"]>=row*156) & 
                                (gene_coords["y"] < (row+1)*156)]
        counts_per_spot = gene_subs["gene"].value_counts()
        counts_per_spot_df = pd.DataFrame(counts_per_spot).transpose()
        counts_per_spot_df["tile"] = tile_name #labels each pseudo-spot with the name of the corresponding image tile.
        gene_tile_df = pd.concat([gene_tile_df, counts_per_spot_df], ignore_index=True) #adds gene counts per pseudo-spot to the gene count dataset

	center_x = (col*156)+78
        center_y = (row*156)+78
        new_row = pd.DataFrame([[tile_name, center_x, center_y]], columns=["tile","x", "y"]) #labels the spatial location of each pseudo-spot with the image tile name.
        spatial_tile_df = pd.concat([spatial_tile_df, new_row], ignore_index=True) #adds spatial location of each pseudo-spot to the spatial matrix dataframe.
#------------------------------------------------------------------------------


#Save the gene counts and spatial location of all pseudo-spot matrices
#------------------------------------------------------------------------------
spatial_tile_df.set_index("tile", inplace=True)
gene_tile_df.set_index("tile", inplace=True)
gene_tile_df.fillna(0, inplace=True)

#Save spatial and gene dataframe
gene_tile_df.to_csv(load_data_dir+"Raw_Gene_Matrix_156.csv", header=True, index=True) #adjust the filename to your needs.
spatial_tile_df.to_csv(load_data_dir+"Raw_Spatial_Matrix_156.csv", header=True, index=True) #adjust the filename to your needs.
#------------------------------------------------------------------------------

#The outputs of this codefile are: image tiles of one H&E WSI in a pre-specified folder. Pseudo-spot transformed gene count and spatial location dataset derived from the in situ sequencing dataset.
#The gene count dataset has the columns "tile" (matching the name of the image tile) and all genes of the in situ sequencing dataset and each row features the number of counts per gene.
#The spatial location dataset has the columns "tile" (matching the name of the image tile) and "x" and "y" that stand for the center location of each pseudo-spot.
#Both datasets are required to create an AnnData object later for processing with Scanpy and Squidpy.