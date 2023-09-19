#This code file runs the training on 85% of the training dataset NCT-CRC-HE-100K. Here, different backbone architectures can be used. 
#In this project EfficientNet, GoogleNet and Resnet50 architectures were trained. Adjust the "XXX" for proper code usage. It needs the data from the DataLoader_100K.py file.

from DataLoader_100K import *
import pickle


#Creating training log text file
#------------------------------------------------------------------------------
training_log = data_dir+"XXX.txt" #adjust to the desired name of the training_log textfile
with open(training_log, "a") as f:
    f.write(date+ " XXX:") #Adapt to model architecture getting trained
    f.write("\n")
#------------------------------------------------------------------------------


#Split Dataset and apply further transformations to trainingset
#------------------------------------------------------------------------------
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
     
    #Get image and label (i.e. tissue class) tensor from dataset and transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    #Get length of dataset
    def __len__(self):
        return len(self.subset)



#Defining transformations for trainset augmentation. Here, you can play with different data augmentation transformations. These were just the ones used in the project
train_transforms = transforms.RandomApply([transforms.RandomRotation(degrees =
                                                                      (0, 180)),
                                            transforms.RandomHorizontalFlip(
                                                p = 0.75),
                                            transforms.RandomVerticalFlip(
                                                p = 0.75),
                                            transforms.RandomAdjustSharpness(
                                                sharpness_factor = 2, p = 0.75),
                                            transforms.RandomAutocontrast(p = 0.75),
                                            transforms.ColorJitter(hue=(-0.2, 0.2))
                                            ], p = 0.5)

#Apply trainset transformations
train_data = MyDataset(train_set, transform=train_transforms)
val_data = MyDataset(train_set, transform=train_transforms)
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
            #the log_softmax was unintentionally used in the project together with a CrossEntropyLoss, which applies log_softmax itself, so that this line 
            #should be removed from the code and only x should be returned
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


#Set k-fold splits, number of training epochs and batch size
#------------------------------------------------------------------------------
k=5 
#number of folds for crossvalidation, this can also be skipped as this code does not perform "real" crossvalidation. 
#Basically what happens in this project is that the model is trained on 4/5 of the whole training data for five epochs and then for other 4/5 and so on. 
#This means that the model sees all training data at least once and is validated during training on the remaining 1/5. 
#Typical crossvalidation would train on 4/5, then validate on 1/5 and then reset the model parameters for training on the next split. 
#Thus, you can use a plain train-validation-test split, e.g. 0.7, 0.15, 0.15 and train for 25 epochs on the training set with monitoring on the validation set during training. 

splits=KFold(n_splits=k, shuffle=True, random_state=42)
num_epochs = 5 #as stated before, adjust this number when not using crossvalidation
batch_size = 64 #adjustable if you want to perform hyperparameter tuning
#------------------------------------------------------------------------------


#Training and Validation steps
#------------------------------------------------------------------------------
#Basic training function returning train_loss and correct training predictions
def train_epoch(model,device,dataloader,loss_fn,optimizer, epoch, fold):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct

#Basic validation function returning train_loss and correct training predictions
def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
      for images, labels in dataloader:

          images,labels = images.to(device),labels.to(device)
          output = model(images)
          loss=loss_fn(output,labels)
          valid_loss+=loss.item()*images.size(0)
          scores, predictions = torch.max(output.data,1)
          val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct

#Concat training and validation dataset due to using Crossvalidation
dataset = ConcatDataset([train_data, val_data])

#Defining Loss Function
criterion = nn.CrossEntropyLoss()
#------------------------------------------------------------------------------


#General training function
#------------------------------------------------------------------------------
def training(cnn, dataset, criterion, num_epochs, batch_size, data_dir, date):
    #Get model and send model to device
    model = cnn
    model.to(device) #device defined in the beginning
    
    #Defining gradient function: Here, different learning rates were used for the backbone (only fine-tuning) and for the added layers. 
    #In general, backbone layers can also be frozen here, or different learning rates could be used.
    optimizer = optim.Adam([
        {"params": model.pretrained.parameters(), "lr": 0.0003},
        {"params": model.my_new_layers.parameters(), "lr": 0.001},
    ])
    
    #Defining training and validation history dictionary 
    history = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}
    valid_loss_min = np.Inf
    batch_train_history = {'epoch': [], 'fold': [], 'batch': [], 'train_loss':[],'train_acc':[]}
    
    #Iterate through epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        
        epoch_to_print = "Epoch{}/{}".format(epoch+1, num_epochs)
        with open(training_log, "a") as f:
                f.write(epoch_to_print)
                f.write("\n")
        
        #Train and validate k folds per epoch
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            #Load data into Dataloader
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
            
            #Training
            training_loss, training_correct=train_epoch(model,device,train_loader,criterion,optimizer,epoch,fold)
            #Validation
            valid_loss, valid_correct=valid_epoch(model,device,val_loader,criterion)

            train_loss = training_loss / len(train_loader.sampler)
            train_acc = training_correct / len(train_loader.sampler) * 100
            val_loss = valid_loss / len(val_loader.sampler)
            val_acc = valid_correct / len(val_loader.sampler) * 100
            
            #Save model parameters if validation loss is improving
            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                
                model_name = pretrained_model +".pt" 
                model_save = data_dir+date+"XXX"+model_name #adapt these two lines of codes to the desired name for the saved model parameters
                
                torch.save(model.state_dict(), model_save)
                print('Improvement-Detected, Saving model')
            
            #Write epoch loss and accuracy to the created training log textfile
            log_to_print = "Fold{}: AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} %".format(
                fold + 1, train_loss, val_loss, train_acc, val_acc)
            with open(training_log, "a") as f:
                f.write(log_to_print)
                f.write("\n")
            
            #Add training and validation loss and accuracy to the history dictionary
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
    return history
#------------------------------------------------------------------------------


#Training with time logging
#------------------------------------------------------------------------------
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

#Calling training function
history, batch_train_history = training(get_model(pretrained_model), dataset, criterion, num_epochs, batch_size, data_dir, date)

end.record()
torch.cuda.synchronize()
train_time = start.elapsed_time(end)
train_time = train_time/60000
print("Training time in min: " + str(train_time))

#Write training time to trainlog textfile
with open(training_log, "a") as f:
    f.write("Training time in min: " + str(train_time))
    f.write("\n")
    f.write("--------------------------------")
    f.write("\n")
    
#Saving training time in Pickle file (if desired, not necessary)
train_time_save_name = data_dir + date + XXX + "train_time.pkl" #adjust to desired output name of training time per model
with open(train_time_save_name, 'wb') as f: 
    pickle.dump([train_time], f)
#------------------------------------------------------------------------------


#Save training history to a .csv file
#------------------------------------------------------------------------------
history_df = pd.DataFrame.from_dict(history, orient="columns")
num_train_episodes = []
for i in range(1, ((k*num_epochs)+1)):
    num_train_episodes.append(i)
history_df["train_episodes"] = num_train_episodes

save_name = data_dir + date + XXX + "_trainhistory.csv" #adjust to desired output name of training history per model
history_df.to_csv(save_name, index=False)
#------------------------------------------------------------------------------


#Plotting Training and Validation curves
#------------------------------------------------------------------------------
def plot_train_curves(history, metric_of_interest): #metric_of_interest "loss" or "accuracy"
    xi = [5,10,15,20,25] #Adapt x-ticks to amount of training episodes
    plt.figure(figsize=(10,8))
    
    if metric_of_interest == "loss": #Plot for loss
        plt.plot(history["train_episodes"], history["train_loss"], label = "Train", marker="o")
        plt.plot(history["train_episodes"], history["val_loss"], label = "Validation", marker="o")
        plt.ylabel("Loss [n.a]", fontsize=12)
        plt.title("Extended " + (pretrained_model).capitalize() + " - Loss", fontsize=16)
        x1,x2,y1,y2 = plt.axis()  
        plt.axis((x1,x2,0,y2)) #Setting y-axis range to start from 0
    elif metric_of_interest == "accuracy": #Plot for accuracy
        plt.plot(history["train_episodes"], history["train_acc"], label = "Train", marker="o")
        plt.plot(history["train_episodes"], history["val_acc"], label = "Validation", marker="o")
        plt.ylabel("Accuracy [%]", fontsize=12)
        plt.title("Extended " + (pretrained_model).capitalize() + " - Accuracy", fontsize=16)
        x1,x2,y1,y2 = plt.axis()  
        plt.axis((x1,x2,y1,100)) #Setting y-axis range to end at 100
        
    plt.xlabel("Training episodes (5 Folds / Epoch)", fontsize=12)
    #plt.xticks(xi)
    plt.axvspan(0.5, 5.5, label= "Epoch 1", color="blue", alpha=0.05) #Adding background colour for each epoch
    plt.axvspan(5.5,10.5, label = "Epoch 2", color="green", alpha=0.05)
    plt.axvspan(10.5,15.5, label = "Epoch 3", color="yellow", alpha=0.05)
    plt.axvspan(15.5,20.5, label = "Epoch 4", color="orange", alpha=0.05)
    plt.axvspan(20.5,25.5, label = "Epoch 5", color="red", alpha=0.05)
    plt.margins(x=0)
    plt.legend(fontsize=11)
    plt.xticks(xi, fontsize=11)
    plt.yticks(fontsize=11)
    #plt.show()
    plt.savefig(data_dir+XXX) #adjust to desired output name of training loss / accuracy plots per model

#Calling plotting functions for training history 
plot_train_curves(history_df, "loss")
plot_train_curves(history_df, "accuracy")
#------------------------------------------------------------------------------

#Outputs of this codefile are: training log textfile, model parameters of model with lowest validation loss during training, training history as .csv file, training time as .pkl file, 
#and training loss and accuracy plots
