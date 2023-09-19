#This code file runs the testing on the external test dataset CRC-VAL-HE-7K. Here, the different trained models can be loaded and tested. 
#Adjust the "XXX" for proper code usage. This file requires the data from the DataLoader_7K.py file as well as the save model parameters from the Train_100K.py file.

from DataLoader_7K import *
import pickle


#Define the model as created in the Train_100K codefile
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

#This parts adds additional layers to the backbone architecture (i.e. EfficientNet + more layers, GoogleNet + more layers, ...)
    class MyNet(nn.Module):
        def __init__(self, my_pretrained_model):
            super(MyNet, self).__init__()
            self.pretrained = my_pretrained_model
            self.my_new_layers = nn.Sequential(nn.Linear(1000, 100),
                                                nn.ReLU(),
                                                nn.BatchNorm1d(100),
                                                nn.Dropout(0.2),
                                                nn.Linear(100, 20),
                                                nn.ReLU(),
                                                nn.Linear(20,9)) #choose x in nn.Linear(20,x) depending on n_classes
        def forward(self, x):
            x = self.pretrained(x)
            x = self.my_new_layers(x)
            output = nn.functional.log_softmax(x, dim=1) 
            #the log_softmax was unintentionally used in the project together with a CrossEntropyLoss, which applies log_softmax itself, 
            #so that this line should be removed from the code and only x should be returned
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


#Define Batch size
#------------------------------------------------------------------------------
batch_size = 64 #define the same batch size as for training
#------------------------------------------------------------------------------


#Test Function
#------------------------------------------------------------------------------
def test(cnn, dataset, batch_size, class_names):   
    #Transformation for Un-Normalizing image tiles
    unorm = transforms.Normalize(mean=[-0.7406/0.1651, -0.5331/0.2174, -0.7059/0.1574],
                                 std=[1/0.1651, 1/0.2174, 1/0.1574])
    
    #Load test dataset into Dataloader
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    #Load model parameters of best performance in training
    best_model = cnn
    best_model.load_state_dict(torch.load(data_dir+"XXX.pt")) #adapt to file.pt and directory with best trained parameters of model
    best_model.to(device)
    best_model.eval()
    
    test_correct = 0
    test_total = 0
    confusion_matrix = np.zeros((len(class_names), len(class_names)))
    
    #Make predictions from test images
    with torch.no_grad():
        for images_t, labels_t in test_loader:
            images_t, labels_t = images_t.to(device), labels_t.to(device)
            output_t = best_model(images_t)
            scores, predictions_t = torch.max(output_t.data,1)
            test_correct+=(predictions_t == labels_t).sum().item()
            test_total += labels_t.size(0)
            
            #Constructing confusion matrix
            for t, p in zip(labels_t.view(-1), predictions_t.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
            
            #Saving wrongly predicted images
            pred_t = output_t.argmax(dim=1, keepdim=True)
            wrong_idx = (pred_t != labels_t.view_as(pred_t)).nonzero()[:, 0]
            wrong_samples = images_t[wrong_idx]
            wrong_preds = pred_t[wrong_idx]
            actual_preds = labels_t.view_as(pred_t)[wrong_idx]
            for i in range(len(wrong_idx)):
                      sample = wrong_samples[i]
                      wrong_pred = wrong_preds[i]
                      actual_pred = actual_preds[i]
                      sample = unorm(sample)
                      img = TF.to_pil_image(sample)
                      img.save('/XXX/wrong_idx{}_pred{}_actual{}.tif'.format(
                          wrong_idx[i], wrong_pred.item(), actual_pred.item()))
                          #Adapt File directory for desired folder           
    
    #Calculate Testset accuracy
    test_accuracy = test_correct / test_total *100
    
    #Confusion Matrix dataframe for absolute numbers
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    
    #Normalized Confusion Matrix with recall values
    df_cm_norm = pd.DataFrame(confusion_matrix/np.sum(confusion_matrix, axis=1)[:, None], 
                              index=class_names, columns=class_names)
    
    #Confusion Matrix with precision values
    df_cm_prec = pd.DataFrame(confusion_matrix/np.sum(confusion_matrix, axis=0)[:, None], 
                              index=class_names, columns=class_names)
    
    
    return test_accuracy, df_cm, df_cm_norm, df_cm_prec
#------------------------------------------------------------------------------


#Testing
#------------------------------------------------------------------------------
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

#Writing Test_log
test_log = data_dir+"XXX.txt" #adjust to desired output name.
with open(test_log, "a") as f:
  f.write("Testing " + pretrained_model + " on " + date + ":")
  f.write("\n")

#Calling Test Function        
t_acc, df_cm, df_cm_norm, df_cm_prec = test(get_model(pretrained_model), test_data_7K, batch_size, test_data_7K.classes)

end.record()
torch.cuda.synchronize()
test_time = start.elapsed_time(end)
test_time = test_time/1000
print("Testing time in s: " + str(test_time))

#Writing Test Time and global test accuracy to testing log textfile
with open(test_log, "a") as f:
    f.write("Testing time in s: " + str(test_time))
    f.write("\n")
    f.write("Global Test Accuracy: " + str(t_acc))
    f.write("\n")
    
#Saving Test time as pickle
test_time_save_name = data_dir + date + "XXX.pkl" #adjust to desired output name of testing time per model
with open(test_time_save_name, 'wb') as f: 
    pickle.dump([test_time], f)
#------------------------------------------------------------------------------


#Saving Confusion-Matrices as csv
#------------------------------------------------------------------------------
save_name_df_cm = data_dir + date + XXX + "_CM_absolutevalues.csv" #adjust to desired output name. Returns .csv file with confusion matrix of absolute numbers of correct and wrong predictions per class
df_cm.to_csv(save_name_df_cm, header=True, index=True)

save_name_df_cm_norm = data_dir + date + XXX + "_CM_normalized.csv" #adjust to desired output name. Returns .csv file with confusion matrix of recall values of correct and wrong predictions per class
df_cm_norm.to_csv(save_name_df_cm_norm, header=True, index=True)

save_name_df_cm_prec = data_dir + date + XXX + "_CM_precision.csv" #adjust to desired output name. Returns .csv file with confusion matrix of precision values of correct and wrong predictions per class
df_cm_prec.to_csv(save_name_df_cm_prec, header=True, index=True)
#------------------------------------------------------------------------------


#Plotting Confusion-Matrices
#------------------------------------------------------------------------------
def plot_cm(cm_data):
    plt.figure(figsize = (15,10))
    cm_heatmap = sns.heatmap(cm_data, cmap = "Blues", annot=True, fmt=",d") #for absolute values no decimals, "," separator for thousands
    cm_heatmap.yaxis.set_ticklabels(cm_heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=11)
    cm_heatmap.xaxis.set_ticklabels(cm_heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=11)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.title("Extended " + pretrained_model.capitalize() + " - Confusion Matrix with absolute numbers", fontsize=16)
    plt.savefig(data_dir+"/Figures/"+date+"_"+pretrained_model+"XXX.tif") #adjust to desired output name.

#Plots decimals (e.g. 0.1234). Used for precision and recall confusion matrices.  
def plot_cm_decimal(cm_data):
    plt.figure(figsize = (15,10))
    cm_heatmap = sns.heatmap(cm_data, cmap = "Blues", annot=True, fmt=".4f") #for decimal values with "." separator
    cm_heatmap.yaxis.set_ticklabels(cm_heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=11)
    cm_heatmap.xaxis.set_ticklabels(cm_heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=11)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    
    if cm_data.equals(df_cm_norm):
        plt.title("Extended " + pretrained_model.capitalize() + " - Normalized Confusion Matrix (Recall)", fontsize=16)
        plt.savefig(data_dir+"/Figures/"+date+"_"+pretrained_model+"XXX.tif") #adjust to desired output name.
    elif cm_data.equals(df_cm_prec):
        plt.title("Extended " + pretrained_model.capitalize() + " - Confusion Matrix (Precision)", fontsize=16)
        plt.savefig(data_dir+"/Figures/"+date+"_"+pretrained_model+"XXX.tif") #adjust to desired output name.

#Calling plot functions        
plot_cm(df_cm)
plot_cm_decimal(df_cm_norm)
plot_cm_decimal(df_cm_prec)
#------------------------------------------------------------------------------


#Getting Precision, Recall, F1-Score per Class from confusion matrix (loaded from csv or directly from test)
#------------------------------------------------------------------------------
#Defining Metrics function for Recall, Precision and F1-Score
def get_metrics(confusion_matrix, classes_list):
    recall = {}
    precision = {}
    f1_score = {}
    
    sum_rows = list(confusion_matrix.sum(axis=1))
    sum_columns = list(confusion_matrix.sum(axis=0))
    
    for i in range(len(classes_list)):
        recall[classes_list[i]] = confusion_matrix[classes_list[i]][i] / sum_rows[i]
        precision[classes_list[i]] = confusion_matrix[classes_list[i]][i] / sum_columns[i]
        f1_score[classes_list[i]] = 2 * 1/((1/(confusion_matrix[classes_list[i]][i] / sum_rows[i]))
                                  +(1/(confusion_matrix[classes_list[i]][i] / sum_columns[i])))
        
    return recall, precision, f1_score

#Calling Metrics function
rec, prec, f1 = get_metrics(df_cm, test_data_7K.classes)

#Create Dataframe of metrics to save as csv
df_rec = pd.DataFrame(
    [{"Classes": classes, "Recall": recall} for (classes), recall in rec.items()])
df_prec = pd.DataFrame(
    [{"Classes": classes, "Precision": precision} for (classes), precision in prec.items()])
df_f1 = pd.DataFrame(
    [{"Classes": classes, "F1-Score": f1_score} for (classes), f1_score in f1.items()])

df_metrics = pd.concat([df_rec, df_prec, df_f1], axis=1, join="outer")
df_metrics = df_metrics.loc[:, ~df_metrics.columns.duplicated()]
df_metrics = df_metrics.set_index("Classes", drop=True)

save_name_df_metrics = data_dir + XXX + "_Metrics.csv" #adjust to desired output name.
df_metrics.to_csv(save_name_df_metrics, header=True, index=True)

#Calculcating and saving Macro-Recall, Macro-Precision and Macro-F1-Score
macro_rec = df_metrics.sum(axis=0)[0]/9
macro_prec = df_metrics.sum(axis=0)[1]/9
macro_f1 = df_metrics.sum(axis=0)[2]/9

#writes metrics to the testing log texfile
with open(test_log, "a") as f:
    f.write("Macro-Recall / Balanced Accuracy: " + str(macro_rec))
    f.write("\n")
    f.write("Macro-Precision: " + str(macro_prec))
    f.write("\n")
    f.write("Macro-F1-Score: " + str(macro_f1))
    f.write("\n")
    f.write("--------------------------------")
    f.write("\n")
    
#Saving Macro-Recall, Macro-Precision and Macro-F1-Score into pickle file
macro_metrics_save_name = data_dir + XXX + "macro_metrics.pkl" #adjust to desired output name.
with open(macro_metrics_save_name, 'wb') as f: 
    pickle.dump([macro_rec, macro_prec, macro_f1], f)
#------------------------------------------------------------------------------

#Outputs of this codefile are: testing log textfile, wrongly predicted image tiles, confusion matrices (absolute numbers, recall, precision) as .csv files, confusion matrices plots, 
#testing time as .pkl file, Precision, recall and F1-score in a .pkl file and a .csv file
