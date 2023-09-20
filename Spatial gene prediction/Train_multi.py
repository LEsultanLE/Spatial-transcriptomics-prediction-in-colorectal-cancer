#This codefile runs the training of one specific CNN model architecture (e.g. EfficientNet, GoogleNet, Resnet50) with three different batch sizes and learning rates.
#The models are trained to predict specified multiple genes from DataLoader_Train_multi.py. MSE loss and Pearson's r are monitored during training.
#The best performing model parameters regarding loss and correlation are saved so that they can be validated on the validation dataset again.
#By that, the models with the best performing hyperparameter setup can be selected for final testing.
#This code requires the CRC tissue classifier model parameters for transfer learning and the loaded data from the Dataloader_Train_multi.py.
#It is important to adapt the load and save directories to your plans. You have to adapt everything that is labelled with XXX.

from Dataloader_Train_multi import *
import pickle


#Set batchsize and LR lists
#------------------------------------------------------------------------------
batch_list = [16, 32, 64] #different batch sizes that were used for hyperparameter tuning, adjustable.
learn_list = ["LLR", "NLR", "HLR"] #different learning rates (s. below) that were used for hyperparameter tuning, adjustable.
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


#Set number of training epochs
#------------------------------------------------------------------------------
num_epochs = 30
#------------------------------------------------------------------------------


#Defining loss function
#------------------------------------------------------------------------------
criterion = nn.MSELoss()
#------------------------------------------------------------------------------


#Training and Validation epoch
#Adapt to the amount of genes that are predicted simultaneously (in the present case it was 4 genes)
#------------------------------------------------------------------------------
def train_epoch(model,device,dataloader,criterion,optimizer):
    train_loss=0.0
    batch_corr_train = 0.0
    model.train()
    for images, gene1_values, gene2_values, gene3_values, gene4_values, path in dataloader:

        images,gene1_values, gene2_values,gene3_values,gene4_values,path = images.to(device),gene1_values.to(device),gene2_values.to(device),gene3_values.to(device),gene4_values.to(device),path
        images, gene1_values, gene2_values, gene3_values, gene4_values = images.float(), gene1_values.float(), gene2_values.float(), gene3_values.float(), gene4_values.float()
        gene1_values = gene1_values.reshape((gene1_values.shape[0],1))
        gene2_values = gene2_values.reshape((gene2_values.shape[0],1))
        gene3_values = gene3_values.reshape((gene3_values.shape[0],1))
        gene4_values = gene4_values.reshape((gene4_values.shape[0],1))
        optimizer.zero_grad()
        g1_output, g2_output, g3_output, g4_output = model(images)
        loss_g1 = criterion(g1_output,gene1_values)
        loss_g2 = criterion(g2_output,gene2_values)
        loss_g3 = criterion(g3_output,gene3_values)
        loss_g4 = criterion(g4_output,gene4_values)
        loss = (loss_g1 + loss_g2 + loss_g3 + loss_g4)/4
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        corr_g1 = stats.pearsonr(g1_output[:,0].cpu().detach().numpy(), gene1_values[:,0].cpu().detach().numpy())[0]
        corr_g2 = stats.pearsonr(g2_output[:,0].cpu().detach().numpy(), gene2_values[:,0].cpu().detach().numpy())[0]
        corr_g3 = stats.pearsonr(g3_output[:,0].cpu().detach().numpy(), gene3_values[:,0].cpu().detach().numpy())[0]
        corr_g4 = stats.pearsonr(g4_output[:,0].cpu().detach().numpy(), gene4_values[:,0].cpu().detach().numpy())[0]
        corr = (corr_g1+corr_g2+corr_g3+corr_g4)/4
        batch_corr_train += corr

    return train_loss, batch_corr_train

def valid_epoch(model,device,dataloader,criterion):
    valid_loss = 0.0
    batch_corr_val = 0.0
    model.eval()
    with torch.no_grad():
      for images, gene1_values, gene2_values, gene3_values,gene4_values, path in dataloader:

          images,gene1_values, gene2_values,gene3_values,gene4_values,path = images.to(device),gene1_values.to(device),gene2_values.to(device),gene3_values.to(device),gene4_values.to(device),path
          images, gene1_values, gene2_values, gene3_values, gene4_values = images.float(), gene1_values.float(), gene2_values.float(), gene3_values.float(), gene4_values.float()
          gene1_values = gene1_values.reshape((gene1_values.shape[0],1))
          gene2_values = gene2_values.reshape((gene2_values.shape[0],1))
          gene3_values = gene3_values.reshape((gene3_values.shape[0],1))
          gene4_values = gene4_values.reshape((gene4_values.shape[0],1))
          g1_output, g2_output, g3_output, g4_output = model(images)
          loss_g1 = criterion(g1_output,gene1_values)
          loss_g2 = criterion(g2_output,gene2_values)
          loss_g3 = criterion(g3_output,gene3_values)
          loss_g4 = criterion(g4_output,gene4_values)
          loss = (loss_g1 + loss_g2 + loss_g3 + loss_g4)/4
          valid_loss+=loss.item()
          
          corr_g1 = stats.pearsonr(g1_output[:,0].cpu().detach().numpy(), gene1_values[:,0].cpu().detach().numpy())[0]
          corr_g2 = stats.pearsonr(g2_output[:,0].cpu().detach().numpy(), gene2_values[:,0].cpu().detach().numpy())[0]
          corr_g3 = stats.pearsonr(g3_output[:,0].cpu().detach().numpy(), gene3_values[:,0].cpu().detach().numpy())[0]
          corr_g4 = stats.pearsonr(g4_output[:,0].cpu().detach().numpy(), gene4_values[:,0].cpu().detach().numpy())[0]
          corr = (corr_g1+corr_g2+corr_g3+corr_g4)/4
          batch_corr_val += corr

    return valid_loss, batch_corr_val
#------------------------------------------------------------------------------


#General training function
#------------------------------------------------------------------------------
def training(cnn, train_dataset, val_dataset, num_epochs, batch_size, data_dir, date, lr, criterion):
    #Get model and send model to device
    model = cnn
    # model.to(device) #device defined in the beginning
    
    #Defining gradient function
    if lr == "LLR":
      optimizer = optim.AdamW([{"params": model.pretrained.parameters(), "lr": 0.000005}, {"params": model.my_new_layers.parameters(), "lr": 0.00005}], weight_decay = 0.005)
    elif lr == "NLR":
      optimizer = optim.AdamW([{"params": model.pretrained.parameters(), "lr": 0.00001}, {"params": model.my_new_layers.parameters(), "lr": 0.0001}], weight_decay = 0.005)
    if lr == "HLR":
      optimizer = optim.AdamW([{"params": model.pretrained.parameters(), "lr": 0.0001},{"params": model.my_new_layers.parameters(), "lr": 0.0005}],weight_decay = 0.005)
        
    #Defining a LR scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    #Defining training and validation history dictionary 
    history = {'train_loss': [], 'train_corr': [], 'val_loss': [], 'val_corr': []}
    valid_loss_min = np.Inf
    valid_corr_max = np.NINF
    
    #Iterate through epochs
    for epoch in range(num_epochs):
        print('Epoch {} / {}:'.format(epoch + 1, num_epochs))
        
        epoch_to_print = "Epoch {} / {}:".format(epoch+1, num_epochs)
        with open(training_log, "a") as f:
                f.write(epoch_to_print)
                f.write("\n")
            
        #Load data into Dataloader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            
        #Training
        training_loss,training_corr = train_epoch(model,device,train_loader,criterion,optimizer)
        #Validation
        valid_loss, valid_corr = valid_epoch(model,device,val_loader,criterion)

        train_loss = (training_loss / len(train_loader.dataset))*1000
        val_loss = (valid_loss / len(val_loader.dataset))*1000
        train_corr = training_corr/len(train_loader)
        val_corr = valid_corr/len(val_loader)
                        
        #Save model parameters if validation loss is improving
        if val_loss < valid_loss_min:
            valid_loss_min = val_loss
                
            model_name = pretrained_model +".pt"
            model_save = data_dir+date+"XXX"+model_name #adapt to desired name for saved model parameters
                
            torch.save(model.state_dict(), model_save)
            print('Loss Improvement-Detected, Saving model')
            
            with open(training_log, "a") as f:
                f.write("Loss Improvement Detected, saving model.")
                f.write("\n")
                
        if val_corr > valid_corr_max:
            valid_corr_max = val_corr
                
            model_name = pretrained_model +".pt"
            model_save = data_dir+date+"XXX"+model_name #adapt to desired name for saved model parameters
                
            torch.save(model.state_dict(), model_save)
            print('Correlation Improvement-Detected, Saving model')
            
            with open(training_log, "a") as f:
                f.write("Correlation Improvement Detected, saving model.")
                f.write("\n")
            
        #Print loss and accuracy during training and validation    
        print("AVG Training Loss:{:.3f} AVG Training Correlation:{:.3f} AVG Validation Loss:{:.3f} AVG Validation Correlation:{:.3f}".format(train_loss, train_corr, val_loss, val_corr))
            
        #Save training log into text file
        log_to_print = "AVG Training Loss:{:.3f} AVG Training Correlation:{:.3f} AVG Validation Loss:{:.3f} AVG Validation Correlation:{:.3f}".format(train_loss, train_corr, val_loss, val_corr)
        with open(training_log, "a") as f:
            f.write(log_to_print)
            f.write("\n")
            
        #Get training and validation history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_corr'].append(train_corr)
        history['val_corr'].append(val_corr) 
    
    scheduler.step()
              
    return history
#------------------------------------------------------------------------------


#Plotting Training and Validation curves
#------------------------------------------------------------------------------
def plot_train_curves(history, metric_of_interest, batch, lr): #metric_of_interest "loss" or "accuracy"
    xi = [5,10,15,20,25,30] #Adapt x-ticks to amount of training episodes
    plt.figure(figsize=(10,8))
    
    if metric_of_interest == "loss": #Plot for loss
        plt.plot(history["train_episodes"], history["train_loss"], label = "Train", marker=".")
        plt.plot(history["train_episodes"], history["val_loss"], label = "Validation", marker=".")
        plt.ylabel("MSE Loss [n.a]", fontsize=12)
        plt.title("Extended " + (pretrained_model).capitalize() + " - Loss", fontsize=16)
        x1,x2,y1,y2 = plt.axis()  
        plt.axis((x1,x2,0,y2)) #Setting y-axis range to start from 0
        
    elif metric_of_interest == "correlation": #Plot for accuracy
        plt.plot(history["train_episodes"], history["train_corr"], label = "Train", marker=".")
        plt.plot(history["train_episodes"], history["val_corr"], label = "Validation", marker=".")
        plt.ylabel("Pearson's r correlation [n.a.]", fontsize=12)
        plt.title("Extended " + (pretrained_model).capitalize() + " - Correlation", fontsize=16)
        x1,x2,y1,y2 = plt.axis()  
        plt.axis((x1,x2,0,1)) #Setting y-axis range to end at 1
        
    plt.xlabel("Training Epoch", fontsize=12)
    #plt.xticks(xi)
    plt.margins(x=0)
    plt.legend(fontsize=11)
    plt.xticks(xi, fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(data_dir+date+"XXX.tif") #adapt to desired save directory
#------------------------------------------------------------------------------


#Training
#------------------------------------------------------------------------------
for batch in batch_list:
  for lr in learn_list:
    data_dir = 'XXX/' #adapt to desired save directory
    
    training_log = data_dir+"XXX.txt" #adapt to desired output name
    with open(training_log, "a") as f:
      f.write(date+ " Resnet50 - multigene:") #Adapt to model backbone architecture (e.g. EfficientNet, GoogleNet ...) getting trained
      f.write("\n")
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
	#adapt everything that is written into the training log textfile to what you want.
      
    model = get_model(pretrained_model)

#Get initial model parameters
    model_dict = model.state_dict()

#Load CRC tissue classifier pretrained model parameters
    pretrained_dict = torch.load("XXX.pt") 
#adapt to file.pt and directory with best trained parameters of model of CRC tissue classifier

#Get rid of the last layers parameters (weights and bias) due to altered model architecture compared to the CRC tissue classifiers
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    pretrained_dict.popitem()
    pretrained_dict.popitem()

#Update initial model parameters with pretrained parameters and load them into model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

#Send model to device (i.e. CUDA)
    model.to(device)
      
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

#Calling training function
    history = training(model, train_data, loaded_valid_dataset, num_epochs, batch, data_dir, date, lr, criterion)

    end.record()
    torch.cuda.synchronize()
    train_time = start.elapsed_time(end)
    train_time = train_time/60000
    print("Training time in min: " + str(train_time))

#Saving training time in Trainlog file
    with open(training_log, "a") as f:
      f.write("Training time in min: " + str(train_time))
      f.write("\n")
      f.write("--------------------------------")
      f.write("\n")
    
#Saving training time in Pickle file
    train_time_save_name = data_dir + date + "XXX.pkl" #adapt to desired save directory
    with open(train_time_save_name, 'wb') as f: 
      pickle.dump([train_time], f)
          
          
    history_df = pd.DataFrame.from_dict(history, orient="columns")
    num_train_episodes = []
    for i in range(1, (num_epochs+1)):
      num_train_episodes.append(i)
    history_df["train_episodes"] = num_train_episodes

    save_name = data_dir + date + "XXX.csv" #adapt to desired save directory
    history_df.to_csv(save_name, index=False)
    
#Plot training MSE loss and Pearson's r correlation curves for training and validation data
    plot_train_curves(history_df, "loss", batch, lr)
    plot_train_curves(history_df, "correlation", batch, lr)
#------------------------------------------------------------------------------

#The output of this codefile are: Model parameters of the best performing model regarding MSE loss and Pearson's during training, training time as .pkl file,
#training log text file with train and validation loss and Pearson's r, training history as .csv file, training history curves plots 
#for each specific hyperparameter setup (e.g. batchsize 16 and LLR (low learning rate)), i.e. 9 times (3 batchsizes x 3 learning rates) in total.