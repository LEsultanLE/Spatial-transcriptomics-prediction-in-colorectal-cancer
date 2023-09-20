#This codefile runs the validation of all created models (from Train_multi.py) from one CNN architecture that is defined in the beginning on one sample and one batch size.
#Thus, you have to run it separately for every model architecture (e.g. EfficientNet, GoogleNet, ...), for every validation sample and every batch size you trained with.
#In the end you get spatial plots and correlation coefficients for every trained model, so that you can choose the best performing model for a single gene.
#This code requires the saved model parameters from Train_multi.py, the loaded data of one sample from the Dataloader_Validation_multi.py and the corresponding H&E WSIs.
#It is important to adapt the load and save directories to your plans. You have to adapt everything that is labelled with XXX.

from Dataloader_Validation_multi import *
import pickle


#Define the model architecture and whether best performing model parameters of validation loss or correlation
#during training should be loaded. In our case correlation was used.
#------------------------------------------------------------------------------
pretrained_model = "resnet50" #adapt to the model architecture that you want to validate
load_metric = "correlation" #adapt to loss if needed
#------------------------------------------------------------------------------


#Set batchsize and LR lists
#------------------------------------------------------------------------------
batch_list = [64] #change batch size. In this case only one batch size was validated in one run of this code.
learn_list = ["LLR", "NLR", "HLR"]
#------------------------------------------------------------------------------


#Creating validation log text file
#------------------------------------------------------------------------------
val_log = data_dir+"XXX.txt" #adapt to desired output name
with open(val_log, "a") as f:
    f.write(date+ " "+pretrained_model+" - multigene:")
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
            self.my_new_layers = nn.Sequential(nn.Linear(1000, 200),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(200),
                                                nn.Dropout(0.3),
                                                nn.Linear(200,10), #choose x in nn.Linear(20,x) depending on n_classes
                                                nn.ReLU(),
                                                nn.Dropout(0.3))
            
#Hard parameter sharing multitask models                                    
            self.gene1 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5,1))
            self.gene2 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5,1))
            self.gene3 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5,1))
            self.gene4 = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5,1))
	#adapt to the amount of genes that are predicted simultaneously, i.e. self.gene5 = nn.Sequential... if there are 5 genes
            
        def forward(self, x):
            x = self.pretrained(x)
            x = self.my_new_layers(x)
            gene1 = self.gene1(x)
            gene2 = self.gene2(x)
            gene3 = self.gene3(x)
            gene4 = self.gene4(x)
            return gene1, gene2, gene3, gene4 #adapt to the amount of genes that are predicted simultaneously
    
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
#Adapt to the amount of genes that are predicted simultaneously (in the present case it was 4 genes)
#------------------------------------------------------------------------------
def validation(gene_list, val_loader, best_model):
  best_model.eval()
  
#Creates a dataframe with gene_true and gene_prediction for every gene that the models should predict
  val_columns = []
  for i in gene_list:
      val_columns.append(i+"_true")
      val_columns.append(i+"_prediction")
  val_columns.insert(0, "tile")

    
  val_df = pd.DataFrame(columns=val_columns)                

#Validation process
  with torch.no_grad():
      for images, gene1_values, gene2_values, gene3_values, gene4_values, path in val_loader:
          images,gene1_values, gene2_values,gene3_values,gene4_values,path = images.to(device),gene1_values.to(device),gene2_values.to(device),gene3_values.to(device),gene4_values.to(device),path
          images, gene1_values, gene2_values, gene3_values, gene4_values = images.float(), gene1_values.float(), gene2_values.float(), gene3_values.float(), gene4_values.float()
          gene1_values = gene1_values.reshape((gene1_values.shape[0],1))
          gene2_values = gene2_values.reshape((gene2_values.shape[0],1))
          gene3_values = gene3_values.reshape((gene3_values.shape[0],1))
          gene4_values = gene4_values.reshape((gene4_values.shape[0],1))
          g1_output, g2_output, g3_output, g4_output = best_model(images)
        
          temp_df = pd.DataFrame()
          c = 0
          temp_df.insert(0, val_columns[c], path)
          c+=1
        
          temp_df.insert(0, val_columns[c], gene1_values[:,0].cpu())
          c+=1
          temp_df.insert(0, val_columns[c], g1_output[:,0].cpu())
          c+=1
          temp_df.insert(0, val_columns[c], gene2_values[:,0].cpu())
          c+=1
          temp_df.insert(0, val_columns[c], g2_output[:,0].cpu())
          c+=1
          temp_df.insert(0, val_columns[c], gene3_values[:,0].cpu())
          c+=1
          temp_df.insert(0, val_columns[c], g3_output[:,0].cpu())
          c+=1
          temp_df.insert(0, val_columns[c], gene4_values[:,0].cpu())
          c+=1
          temp_df.insert(0, val_columns[c], g4_output[:,0].cpu())
        
          temp_df = temp_df[val_columns]  
          val_df = pd.concat([val_df, temp_df])
        
  val_df.reset_index(drop=True, inplace=True) 
  
  return val_df       
#------------------------------------------------------------------------------


#Calculate Pearson's correlation coefficient per sample
#------------------------------------------------------------------------------
def pearson_calc(val_df, true, preds, pearson_dict, batch, lr, sample):
  for j in range(len(true)):
    pearson_name = true[j].replace("true", "pearson")
    pearson_name = pearson_name + "_"+str(batch)+"_"+lr+"_"+sample
    pearson = stats.pearsonr(val_df[true[j]], val_df[preds[j]])
    with open(val_log, "a") as f:
      f.write(pearson_name + ": " + ("%.4f" % pearson[0]))
      f.write("\n")
      f.write("------------------------------")
      f.write("\n")
    pearson_dict[pearson_name] = pearson[0]
#------------------------------------------------------------------------------


#Create spatial plots of validation sample (ground truth and prediction)
#------------------------------------------------------------------------------
def spatial_plot(val_df, true, preds, batch, lr, sample): 
#Load spatial matrix
  spatial_matrix = pd.read_csv("/XXX/"+sample+".csv") #adapt to directory of spatial matrix dataset (from Preprocessing_Tiling.py)
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
  h_e_img = plt.imread("/XXX/"+sample+".tif") #load stain normalized H&E WSI created in Stain_Normalization.py

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
    spatial_plot_name = data_dir+"/XXX/"+sample+".tif" #adapt to desired output name
    sq.pl.spatial_scatter(adata_test, use_raw=False, shape="circle", color=[true[l], preds[l]], title=[true[l].replace("_true", "")+" - groundtruth", preds[l].replace("_prediction", "")+" - prediction"], alpha = 0.8, cmap="bwr", axis_label=["", ""], scalebar_dx=0.72, scalebar_units="um", scalebar_kwargs={"fixed_value": 500, "location": "lower left", "box_alpha": 0.5}, save = spatial_plot_name)
#------------------------------------------------------------------------------


#Running the whole validation process
#------------------------------------------------------------------------------
pearson_dict = {}

for batch in batch_list:
  for lr in learn_list:
    with open(val_log, "a") as f:
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
#adapt everything that is written into the val log textfile to what you want.
    
    load_data_dir = 'XXX' #include the directory of the .pt file with best performing model parameters created from Train_multi.py
    best_model = get_model(pretrained_model)
    best_model.load_state_dict(torch.load(load_data_dir+"XXX.pt")) #adapt to file.pt and directory with best trained parameters of model
    best_model.to(device)
    
    val_loader = DataLoader(loaded_valid_dataset, batch_size=batch, shuffle=False)
    
#applying the validation function
    val_df = validation(gene_list, val_loader, best_model)
    
    true = list(val_df)[1::2]
    preds = list(val_df)[2::2]
   
#Applying Pearson's r correlation function
    pearson_calc(val_df, true, preds, pearson_dict, batch, lr, sample)
    
#Applying spatial plot function  
    spatial_plot(val_df, true, preds, batch, lr, sample)
    
#Create dataframe from Pearson's r correlation and save it as .csv file
pearson_df = pd.DataFrame([pearson_dict])
pearson_df = pearson_df.T

pearson_df_save = data_dir + "XXX"+sample+".csv" #adapt to desired output name
pearson_df.to_csv(pearson_df_save, header=False, index=True)

with open(val_log, "a") as f:
  f.write("--------------------------------------------")
  f.write("\n")
#------------------------------------------------------------------------------

#The outputs of this codefile are: .csv file with pearson's correlation coefficient of one model architecture, one batch size and all learning rates
#for all genes (i.e. 64_LLR_PIGR, 64_LLR_RAMP1, 64_LLR_TNS1, 64_LLR_RUBCNL, 64_NLR_PIGR, 64_NLR_RAMP1, ... for Resnet50).
#The corresponding spatial plots for ground truth and prediction. A val_log .txt file with information on hyperparameters and Pearson's r correlation.