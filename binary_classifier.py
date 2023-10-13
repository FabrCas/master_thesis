from    time                import time
from    tqdm                import tqdm
from    datetime            import date

import  torch               as T
from    torch.nn            import functional as F
from    torch.optim         import Adam, lr_scheduler
from    torch.cuda.amp      import GradScaler, autocast
from    torch.utils.data    import DataLoader

from    utilities           import *
from    dataset             import CDDB_binary
from    models              import ResNet_ImageNet, ResNet

from sklearn.metrics import precision_score, recall_score, f1_score,     \
        confusion_matrix, hamming_loss, jaccard_score, accuracy_score

class DFD_BinClassifier(object):
    """
        binary classifier for deepfake detection using CDDB dataset
    """
    def __init__(self, useGPU = True, batch_size = 32, model_type = "resnet_pretrained"):
        """ init classifier

        Args:
            useGPU (bool, optional): flag to use CUDA device or cpu hardware by the model. Defaults to True.
        """
        super(DFD_BinClassifier, self).__init__()
        self.useGPU = useGPU
        self.batch_size = batch_size
        self.model_type = model_type
        
        # load dataset & dataloader
        self.train_dataset = CDDB_binary(train = True)
        self.test_dataset = CDDB_binary(train = False)
        
        # laod model
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        if model_type == "resnet_pretrained":
            self.model = ResNet_ImageNet(n_classes = 2).getModel()
        else:
            self.model = ResNet()
        self.model.to(self.device)
        
        # define loss and final activation function
        self.sigmoid = F.sigmoid
        self.bce = F.binary_cross_entropy_with_logits
        
        # learning parameters (default)
        self.lr = 1e-5
        self.n_epochs = 20
        self.weight_decay = 0.001       # L2 regularization term 
        
        
    def train(self, name_train):
        
        # define train dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # define the optimization algorithm
        optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        # learning rate scheduler
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        
        # model in training mode
        self.model.train()
        
        # define the gradient scaler to avoid weigths explosion
        scaler = GradScaler()
        
        # intialize data structure to keep track of training performance
        loss_epochs = []
        
        # training loop over epochs
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch{epoch_idx+1}]             \n")
            
            # define cumulative loss for the current epoch
            loss_epoch = 0
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= n_steps):
                
                # if step_idx > 5: break
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(True)
                y = y.to(self.device)               # binary int encoding for each sample
                y = y.to(T.float)
                
                # zeroing the gradient
                optimizer.zero_grad()
                
                # model forward and loss computation
                with autocast():
                    logits = self.model.forward(x)

                    loss = F.binary_cross_entropy_with_logits(input=logits, target=y)
                
                # update total loss    
                loss_epoch += loss.item()
                
                # loss backpropagation
                scaler.scale(loss).backward()
                
                # compute updates using optimizer
                scaler.step(optimizer)

                # update weights through scaler
                scaler.update()
                
                # lr scheduler step 
                scheduler.step()
                
            # compute average loss for the epoch
            avg_loss = loss_epoch/n_steps
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
                
            # break
        
        
        # create paths and file names for saving training outcomes
        current_date = date.today().strftime("%d-%m-%Y")
        
        path_model_folder = os.path.join("./models",  name_train + "_" + current_date)
        path_loss_folder  = os.path.join("./results", name_train + "_" + current_date)
        model_name = str(self.n_epochs) +'.ckpt'
        
        if (not os.path.exists(path_model_folder)):
            os.makedirs(path_model_folder)
        if (not os.path.exists(path_loss_folder)):
            os.makedirs(path_loss_folder)
        
        # save loss plot
        plot_loss(loss_epochs, epoch= self.n_epochs, model_name= name_train, path_save = path_loss_folder)
        
        # save model
        saveModel(self.model, model_name, path_model_folder)
    
    def test(self):
        
        # define test dataloader
        train_dataloader = DataLoader(self.test_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
    
        # model in evaluation mode
        self.model.eval()  
        
        with T.no_grad():
            pass
    
    def metrics(self):
        pass 

#TODO        
class OOD_BinDetector(object):
    """
        Detector for OOD data
    """
    
    def __init__(self):
        pass 
        
# [test section] 
if __name__ == "__main__":
    # dataset = CDDB_binary()
    # test_num_workers(dataset, batch_size  =32)   # use n_workers = 8
    bin_classifier = DFD_BinClassifier()
    bin_classifier.train(name_train="resnet50_ImageNet")