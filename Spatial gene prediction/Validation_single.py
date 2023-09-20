#This codefile runs the validation of all created models (from Train_single.py) from one CNN architecture that is defined in the beginning on one sample.
#Thus, you have to run it separately for every model architecture (e.g. EfficientNet, GoogleNet, ...) and for every sample.
#In the end you get spatial plots and correlation coefficients for every trained model, so that you can choose the best performing model for a single gene.
#This code requires the saved model parameters from Train_single.py and the loaded data of one sample from the Dataloader_Validation_single.py.
#It is important to adapt the load and save directories to your plans. You have to adapt everything that is labelled with XXX.

from Dataloader_Validation_single import *
import pickle


#Define the model architecture and whether best performing model parameters of validation loss or correlation
#during training should be loaded. In our case correlation was used.
#------------------------------------------------------------------------------
pretrained_model = "googlenet"
load_metric = "correlation" #adapt to loss if needed
#------------------------------------------------------------------------------


#Set batchsize and LR lists
#------------------------------------------------------------------------------
batch_list = [16, 32, 64]
learn_list = ["LLR", "NLR", "HLR"]
#------------------------------------------------------------------------------


#Creating validation log text file
#------------------------------------------------------------------------------
validation_log = data_dir+"XXX.txt" #adapt to desired output name
with open(validation_log, "a") as f:
    f.write(date+ " "+pretrained_model+" - "+gene_list[0]+":")
    f.write("\n")
#------------------------------------------------------------------------------


#Import and define Model
#------------------------------------------------------------------------------
#Extendable function for downloading pretrained CNN
pretrained_model = "efficientnet"

def get_model(pretrained_model):
    if pretrained_model == "resnet50":
        pretrained_net = models.resnet50(weights="IMAGENET1K_V2")
    elif pretrained_model == "googlenet":
        pretrained_net = models.googlenet(weights = "IMAGENET1K_V1")
    elif pretrained_model == "efficientnet":
        pretrained_net = models.efficientnet_b0(weights="IMAGENET1K_V1")
        
    #elif pretrained_model == "xxx":
        #pretrained_net = models.xxx(weights="IMAGENET1K_V1") #You can use any other available CNN architecture here!

#Adaptation of pretrained CNN with additional layers output layer neurons 
#matching n_regressions to predict        
    class MyNet(nn.Module):
        def __init__(self, my_pretrained_model):
            super(MyNet, self).__init__()
            self.pretrained = my_pretrained_model
            
            n_output_neurons = len(gene_list) #in the case of the single-gene models, this equals to 1
            
            self.my_new_layers = nn.Sequential(nn.Linear(1000, 200),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(200),
                                                nn.Dropout(0.3),
                                                nn.Linear(200, n_output_neurons))
            

        def forward(self, x):
            x = self.pretrained(x)
            output = self.my_new_layers(x)
            return output
    
    #Initializing adapted CNN based on chosen pretrained architecture
    extended_net = MyNet(my_pretrained_model=pretrained_net)
    
    #Initializing weights for newly appended layers in MyNet
    def initialize_weights_linear(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                y = 1.0/np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0)
                
    extended_net.apply(initialize_weights_linear)
    
    return extended_net
#------------------------------------------------------------------------------


#Validation function
#------------------------------------------------------------------------------
def validation(gene_list, val_loader):
  #create empty dataframes with columns of image tile name, gene_true and gene_prediction
  val_columns = []
  for i in gene_list:
      val_columns.append(i+"_true")
      val_columns.append(i+"_prediction")
  val_columns.insert(0, "tile")

    
  val_df = pd.DataFrame(columns=val_columns)                

#Validation process
  with torch.no_grad():
      for images, labels, path in val_loader:
          images = images.to(device)
          images = images.float()
        
          labels = torch.stack(labels, dim=1)
          labels = labels.to(device)
        
          outputs = best_model(images)
          #outputs = outputs.cpu().numpy()
        
          temp_df = pd.DataFrame()
          c = 0
          temp_df.insert(0, val_columns[c], path)
          c+=1
        
          num_gene = outputs.shape[1]
        
        
          for i in range(num_gene):
              temp_df.insert(0, val_columns[c], labels[:,i].cpu())
              c+=1
              temp_df.insert(0, val_columns[c], outputs[:,i].cpu())
              c+=1
          temp_df = temp_df[val_columns]  
          val_df = pd.concat([val_df, temp_df])
        
  val_df.reset_index(drop=True, inplace=True)
  
  return val_df        
#------------------------------------------------------------------------------


#Calculate Pearson's correlation coefficient per sample
#------------------------------------------------------------------------------
def pearson_calc(val_df, true, preds, pearson_dict, batch, lr):
  for j in range(len(true)):
    pearson_name = true[j].replace("true", "pearson")
    pearson_name = pearson_name + "_"+str(batch)+"_"+lr+"_"+sample
    pearson = stats.pearsonr(val_df[true[j]], val_df[preds[j]])
    with open(validation_log, "a") as f:
      f.write(pearson_name + ": " + ("%.4f" % pearson[0]))
      f.write("\n")
      f.write("-----------------------------------")
      f.write("\n")
    pearson_dict[pearson_name] = pearson[0]
#------------------------------------------------------------------------------


#Create spatial plots of validation sample (ground truth and prediction)
#------------------------------------------------------------------------------
def spatial_plot(val_df, true, preds, batch, lr): 
#Load spatial matrix
  spatial_matrix = pd.read_csv("XXX.csv") #adapt to directory of spatial matrix dataset (from Preprocessing_Tiling.py)
  spatial_matrix['tile'] = spatial_matrix['tile'].apply(lambda x: "{}{}".format(x, ".tiff"))
  spatial_matrix['tile'] = spatial_matrix['tile'].apply(lambda x: "{}{}".format("XXX")) 
#adjust XXX in such a way that the name of the image tile in spatial_matrix exactly matches the name (including directory) of the H&E image tiles. 

#Merge spatial with test dataset to get spatial coordinates for predicted tiles
  merge = pd.merge(val_df, spatial_matrix, on = "tile")

#Get filtered spatial matrix
  new_spatial = merge.copy()
  new_spatial = new_spatial[["tile", "x", "y"]]
  new_spatial.set_index("tile", inplace=True)
  spatial_array = new_spatial.to_numpy()

#Get filtered gene matrix
  new_gene = merge.copy()
  new_gene.drop(columns=["x", "y"], inplace=True)
  new_gene.set_index("tile", inplace=True)

#Load Image
  Image.MAX_IMAGE_PIXELS = None
  h_e_img = plt.imread("XXX.tif") #load stain normalized H&E WSI created in Stain_Normalization.py

#Setup AnnData object from filtered gene_matrix (includes ground truth and predictions), spatial matrix and H&E WSI
  adata_test = AnnData(new_gene, obsm={"spatial": spatial_array})
  adata_test.var
  spatial_key = "spatial"
  library_id = "tissue42"
  adata_test.uns[spatial_key] = {library_id: {}}
  adata_test.uns[spatial_key][library_id]["images"] = {}
  adata_test.uns[spatial_key][library_id]["images"] = {"hires": h_e_img}
  adata_test.uns[spatial_key][library_id]["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 130}

#Plot spatial gene truth and prediction
  for l in range(len(true)):
    spatial_plot_name = data_dir+"XXX.tif" #adapt to desired output name
    sq.pl.spatial_scatter(adata_test, use_raw=False, shape="circle", color=[true[l], preds[l]], title=[true[l].replace("_true", "")+" - groundtruth", preds[l].replace("_prediction", "")+" - prediction"], alpha = 0.8, cmap="bwr", axis_label=["", ""], scalebar_dx=0.72, scalebar_units="um", scalebar_kwargs={"fixed_value": 500, "location": "lower left", "box_alpha": 0.5}, save = spatial_plot_name)
#------------------------------------------------------------------------------


#Running the whole validation process
#------------------------------------------------------------------------------
pearson_dict = {}

for batch in batch_list:
  for lr in learn_list:
    with open(validation_log, "a") as f:
      f.write("Batchsize: " + str(batch)) #adapt to batch size
      f.write("\n")
      f.write("Learning Rate: " + lr) #adapt to learning rate
      f.write("\n")
      f.write("Optimizer: AdamW")
      f.write("\n")
      f.write("Exponential LR scheduler: 0.9")
      f.write("\n")
      f.write("Loss Function: MSE Loss")
      f.write("\n")
      f.write("Metric for parameter saved: " + load_metric)
      f.write("\n")
      f.write("Sample: " + sample)
      f.write("\n")
	#adapt everything that is written into the validation log textfile to what you want.
    
    load_data_dir = 'XXX' #include the directory of the .pt file with best performing model parameters created from Train_single.py
    best_model = get_model(pretrained_model)
    best_model.load_state_dict(torch.load(load_data_dir+"XXX.pt")) #adapt to file.pt and directory with best trained parameters of model
    best_model.to(device)
    
    val_loader = DataLoader(loaded_valid_dataset, batch_size=batch, shuffle=False)
    
#applying the validation function
    val_df = validation(gene_list, val_loader)
    
    true = list(val_df)[1::2]
    preds = list(val_df)[2::2]

#Applying Pearson's r correlation function
    pearson_calc(val_df, true, preds, pearson_dict, batch, lr)
 
#Applying spatial plot function  
    spatial_plot(val_df, true, preds, batch, lr)

#Create dataframe from Pearson's r correlation and save it as .csv file
pearson_df = pd.DataFrame([pearson_dict])

pearson_df_save = data_dir + "XXX.csv" #adapt to desired output name
pearson_df.to_csv(pearson_df_save, header=True, index=False)
#------------------------------------------------------------------------------

#The outputs of this codefile are: .csv file with pearson's correlation coefficient of all validated models per gene and validation sample
#(i.e. Pearson's correlation for EfficientNet with 3 different batchsizes and learning rates for gene X and sample Y)
#, and the corresponding spatial plots for ground truth and prediction.