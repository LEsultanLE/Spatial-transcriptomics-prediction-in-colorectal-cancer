#This codefile runs the testing on one sample and one gene defined in DataLoader_Test_single.py. 
#Thus, if there are e.g. 3 test samples and 6 genes, the code (including the adjusted Dataloader_Test_single.py) needs to be run 18 times.
#This code requires the saved best model parameters from Train_single.py and the loaded data of one sample and gene from the Dataloader_Test_single.py.
#It is important to adapt the load and save directories to your plans. You have to adapt everything that is labelled with XXX.

from Dataloader_Test_single import *
import pickle


#Define whether best performing model parameters of validation loss or correlation
#during training should be loaded. In our case correlation was used.
#------------------------------------------------------------------------------
load_metric = "correlation" #adapt to loss if needed
#------------------------------------------------------------------------------


#Set best performing model architecture and hyperparameters for every trained single gene
#------------------------------------------------------------------------------
param_list = [["resnet50", 32, "LLR"], ["resnet50", 64, "NLR"], ["resnet50", 64, "NLR"], ["resnet50", 64, "HLR"], ["efficientnet", 64, "HLR"], ["efficientnet", 16, "HLR"]]
#In our case gene 1 was best predicted on the validation samples with Resnet50 backbone, batchsize 32 and low learning rate, gene 2 with Resnet50 backbone, btachsize 64
#and normal learning rate, and so on. Thus, you will have to adapt this to your validation results from Validation_single.py.
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


#Define testing function
#------------------------------------------------------------------------------
def testing(gene_list, test_loader):
#create empty dataframes with columns of image tile name, gene_true and gene_prediction
  test_columns = []
  for i in gene_list:
      test_columns.append(i+"_true")
      test_columns.append(i+"_prediction")
  test_columns.insert(0, "tile")

    
  test_df = pd.DataFrame(columns=test_columns)                

#Testing process
  with torch.no_grad():
      for images, labels, path in test_loader:
          images = images.to(device)
          images = images.float()
        
          labels = torch.stack(labels, dim=1)
          labels = labels.to(device)
        
          outputs = best_model(images)
          #outputs = outputs.cpu().numpy()
        
          temp_df = pd.DataFrame()
          c = 0
          temp_df.insert(0, test_columns[c], path)
          c+=1
        
          num_gene = outputs.shape[1]
        
        
          for i in range(num_gene):
              temp_df.insert(0, test_columns[c], labels[:,i].cpu())
              c+=1
              temp_df.insert(0, test_columns[c], outputs[:,i].cpu())
              c+=1
          temp_df = temp_df[test_columns]  
          test_df = pd.concat([test_df, temp_df])
        
  test_df.reset_index(drop=True, inplace=True)
  
  return test_df        
#------------------------------------------------------------------------------


#Pearson function
#------------------------------------------------------------------------------
def pearson_calc(test_df, true, preds, params):
  for j in range(len(true)):
    pearson_name = true[j].replace("true", "pearson")
    pearson_name = pearson_name + "_"+str(params[1])+"_"+params[2]+"_"+sample
    pearson = stats.pearsonr(test_df[true[j]], test_df[preds[j]])
    with open(test_log, "a") as f:
      f.write(pearson_name + ": " + ("%.4f" % pearson[0]))
      f.write("\n")
      f.write("-----------------------------------")
      f.write("\n")
#Define save name. In this case the save name includes the best performing model architecture, batchsize and learning rate as well as the predicted gene.
#Adapt to your needs.
    pearson_save_name = data_dir + date + "_XXX_"+str(params[1])+"_" + params[2] +"_" + gene_list[0]+ params[0] + "pearson.pkl"
    with open(pearson_save_name, 'wb') as f: 
      pickle.dump([pearson], f)
#------------------------------------------------------------------------------


#Create spatial plots of test sample (ground truth and prediction)
#------------------------------------------------------------------------------
def spatial_plot(test_df, true, preds, params): 
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

#Save predicted anndata object
#Define save name. In this case the save name includes the best performing model architecture, batchsize and learning rate as well as the predicted gene.
#Adapt to your needs.
  adata_test.write(data_dir + date + "_Test_"+gene_list[0]+"_" + params[0] + "_" + str(params[1]) + "_" + params[2] + "_" +sample+".h5ad")

#Plot spatial gene truth and prediction
  for l in range(len(true)):
#Define save name. In this case the save name includes the best performing model architecture, batchsize and learning rate as well as the predicted gene.
#Adapt to your needs.
    spatial_plot_name = data_dir+"/Figures/ST_Predict_absolute_single_"+str(params[1])+"_"+params[2]+"_"+gene_list[0]+params[0]+"_test_spatial_"+load_metric+"_"+sample+".tif" 

    sq.pl.spatial_scatter(adata_test, use_raw=False, shape="circle", color=[true[l], preds[l]], title=[true[l].replace("_true", "")+" - groundtruth", preds[l].replace("_prediction", "")+" - prediction"], alpha = 0.8, cmap="bwr", axis_label=["", ""], scalebar_dx=0.72, scalebar_units="um", scalebar_kwargs={"fixed_value": 500, "location": "lower left", "box_alpha": 0.5}, save = spatial_plot_name)
#------------------------------------------------------------------------------


#Define which gene is linked with which best performing architecture and hyperparameter from param_list
#------------------------------------------------------------------------------
if gene_list[0] == "PIGR":
  params = param_list[0]
  load_date = "10082023"
elif gene_list[0] == "TNS1":
  params = param_list[1]
  load_date = "10082023"
elif gene_list[0] == "RUBCNL":
  params = param_list[2]
  load_date = "10082023"
elif gene_list[0] == "RAMP1":
  params = param_list[3]
  load_date = "11082023"
elif gene_list[0] == "COL1A1":
  params = param_list[4]
  load_date = "08082023"
elif gene_list[0] == "COL1A2":
  params = param_list[5]
  load_date = "09082023"
#------------------------------------------------------------------------------


#Running the whole test process
#------------------------------------------------------------------------------
#Define load data directory where .pt file is found
load_data_dir = '/fast/users/jole12_c/work/ST_Predict/Absolute_Pred/Training_Results/'+str(params[1])+'_'+params[2]+'/'

#Define and load the model based on the architecture that was selected in params
best_model = get_model(params[0])
best_model.load_state_dict(torch.load(load_data_dir+load_date+"_"+gene_list[0]+"_ST_absolute_single_"+str(params[1])+"_"+params[2]+"_"+load_metric+"_"+params[0]+".pt")) #adapt to file.pt and directory with best trained parameters of model
best_model.to(device)
best_model.eval()
    
test_loader = DataLoader(loaded_test_dataset, batch_size=params[1], shuffle=False)

#Run Test process
test_df = testing(gene_list, test_loader)

true = list(test_df)[1::2]
preds = list(test_df)[2::2]

#Save the ground truth and predicted values in a .csv file
test_save_name = data_dir + date + "XXX"+str(params[1])+"_" + params[2] +"_" + gene_list[0]+ params[0] + "_data_"+load_metric+"_"+sample+".csv" #adapt to sample
test_df.to_csv(test_save_name, header=True, index=False)

#Applying the correlation function
pearson_calc(test_df, true, preds, params)

#Applying the spatial plot function
spatial_plot(test_df, true, preds, params)
#------------------------------------------------------------------------------


#The outputs of this codefile are: .pkl file with the pearson's correlation coefficient, the spatial plots for ground truth and prediction,
#the AnnData object of ground truth and prediction, and a .csv file of groundtruth and predicted values for one sample and one gene that were
#defined in DataLoader_Test_single.py, and a test_log .txt file with model parameters and Pearson's r correlation.
