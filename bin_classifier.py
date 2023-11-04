from    time                import time
from    tqdm                import tqdm
from    datetime            import date

import  torch               as T
import  numpy               as np
import  os

from    torch.nn            import functional as F
from    torch.optim         import Adam, lr_scheduler
from    torch.cuda.amp      import GradScaler, autocast
from    torch.utils.data    import DataLoader

from    utilities           import plot_loss, saveModel, metrics_binClass, loadModel, test_num_workers, sampleValidSet
from    dataset             import CDDB_binary, CDDB_binary_Partial
from    models              import ResNet_ImageNet, ResNet


# from    sklearn.metrics     import precision_recall_curve, auc, roc_auc_score


"""
                        Binary Deepfake classifiers trained on CDDB dataset with different presets
"""



class BinaryClassifier(object):
    
    def __init__(self, useGPU, batch_size, model_type):
        super(BinaryClassifier, self).__init__()
        self.model_type     = model_type
        self.useGPU         = useGPU
        self.batch_size     = batch_size
        
        # path 2 save
        self.path_models    = "./models/bin_class"
        self.path_results   = "./results/bin_class"
        
        
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        self.modelEpochs = 0        # variable used to store the actual number of epochs used to learn the model
        
        # initialize None variables
        self.model = None
    
    def init_weights_normal(self):
        print("Initializing weights using Gaussian distribution")
        # Initialize the weights with Gaussian distribution
        for param in self.model.parameters():
            if len(param.shape) > 1:
                T.nn.init.normal_(param, mean=0, std=0.01) 
                
    def init_weights_kaimingNormal(self):
        # Initialize the weights  using He initialization
        print("Weights initialization using kaiming Normal")
        for param in self.model.parameters():
            if len(param.shape) > 1:
                T.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    
    def getLayers(self, show = True):
        if not self.model is None: 
            layers = dict(self.model.named_parameters())
            dtype = "unknown"
            for k,v in layers.items():
                print("name: {:<30}, shape layer: {:>20}".format(k, str(list(v.data.shape)), str(v.dtype)))
                if dtype == "unknown": dtype = str(v.dtype)
            print (f"The model's parameters data are of type: {dtype}")
            return layers
        else: print("model has been not defined")
        
    def forward(self, x):
        """ network forward

        Args:
            x (T.Tensor): input image/images

        Returns:
            pred: label: 0 -> real, 1 -> fake
        """
        if self.model is None: raise ValueError("No model has been defined, impossible forwarding the data")
        
        if not(isinstance(x, T.Tensor)):
            x = T.tensor(x)
        
        # handle single image, increasing dimensions for batch
        if len(x.shape) == 3:
            x = T.expand(1,-1,-1,-1)
        elif len(x.shape) <= 2 or len(x.shape) >= 5:
            raise ValueError("The input shape is not compatiple, expected a batch or a single image")
        
        # correct the dtype
        if not (T.dtype is T.float32):
            x = x.to(T.float32)
         
        x = x.to(self.device)
        
        logits      = self.model.forward(x)
        probs       = self.sigmoid(logits)
        pred        = T.argmax(probs, -1)
        fake_prob   = probs[:,1]   # positive class probability (fake probability)
        
        return pred, fake_prob, logits
    
    
    def load(self, folder_model, epoch):
        try:
            self.path2model         = os.path.join(self.path_models,  folder_model, str(epoch) + ".ckpt")
            self.path2model_results = os.path.join(self.path_results, folder_model)
            self.modelEpochs         = epoch
            loadModel(self.model, self.path2model)
            self.model.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except:
            print("No model: {} found for the epoch: {} in the folder: {}".format(folder_model, epoch, self.path_models))
            
    def test(self):
        
        # define test dataloader
        test_dataloader = DataLoader(self.test_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)

        # compute number of batches for test
        n_steps = len(test_dataloader)
        print("Number of batches for test: {}".format(n_steps))
        
        # model in evaluation mode
        self.model.eval()  
        
        # define the array to store the result
        predictions             = np.empty((0), dtype= np.int16)
        predicted_probabilities = np.empty((0), dtype= np.float32)
        targets                 = np.empty((0), dtype= np.int16)
        
        T.cuda.empty_cache()
        
        # loop over steps
        for step_idx,(x,y) in tqdm(enumerate(test_dataloader), total= n_steps):
            
            # prepare samples/targets batches 
            x = x.to(self.device)
            y = y.to(self.device)               # binary int encoding for each sample
            y = y.to(T.float)
            target = T.argmax(y, dim =-1).cpu().numpy().astype(int)
            
            with T.no_grad():
                with autocast():
                    pred, prob, _ = self.forward(x)
                    pred = pred.cpu().numpy().astype(int)
                    prob = prob.cpu().numpy()
            
            predictions             = np.append(predictions, pred, axis  =0)
            predicted_probabilities = np.append(predicted_probabilities, prob, axis  =0)
            targets                 = np.append(targets,  target, axis  =0)
            
                
        # create folder for the results
        if (not os.path.exists(self.path2model_results)):
            os.makedirs(self.path2model_results)
            
        # compute metrics from test data
        metrics_binClass(predictions, targets, predicted_probabilities, epoch_model= str(self.modelEpochs), path_save = self.path2model_results)
            
class DFD_BinClassifier_v1(BinaryClassifier):
    """
        binary classifier for deepfake detection using CDDB dataset, first version
        including usage of ResNet model, no validation set used during training (70% training, 30% testing),
        no data augementation.
        
        training model folders:
        models/bin_class/resnet50_ImageNet_13-10-2023
    """
    def __init__(self, useGPU = True, batch_size = 32, model_type = "resnet_pretrained"):
        """ init classifier

        Args:
            useGPU (bool, optional): flag to use CUDA device or cpu hardware by the model. Defaults to True.
        """
        super(DFD_BinClassifier_v1, self).__init__(useGPU = useGPU, batch_size = batch_size, model_type = model_type)
            
        # load dataset & dataloader
        self.train_dataset = CDDB_binary(train = True)
        self.test_dataset = CDDB_binary(train = False)
        
        # load model
        if self.model_type == "resnet_pretrained":   # fine-tuning
            self.model = ResNet_ImageNet(n_classes = 2).getModel()
            self.path2model_results   = os.path.join(self.path_results, "ImageNet")     
        else:                                   # training from scratch
            self.model = ResNet()
            self.init_weights_kaimingNormal()    # initializaton of the wights
            self.path2model_results   = os.path.join(self.path_results, "RandomIntialization")
            
        self.model.to(self.device)
        
        # define loss and final activation function
        self.sigmoid = F.sigmoid
        self.bce     = F.binary_cross_entropy_with_logits
        
        # learning parameters (default)
        self.lr             = 1e-5
        self.n_epochs       = 20
        self.weight_decay   = 0.001       # L2 regularization term 
        
        
        
    def valid(self):
        raise NotImplementedError
    
    def train(self, name_train):
        
        # define train dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # can be included the valid dataloader, look for sampleValidSet in utilities.py module
        
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
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
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

                    loss = self.bce(input=logits, target=y)   # logits bce version
                
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
            
            # include validation here if needed
                
            # break
        
        
        # create paths and file names for saving training outcomes
        current_date = date.today().strftime("%d-%m-%Y")
        
        path_model_folder       = os.path.join(self.path_models,  name_train + "_" + current_date)
        name_model_file         = str(self.n_epochs) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        path_results_folder     = os.path.join(self.path_results, name_train + "_" + current_date)
        name_loss_file          = 'loss_'+ str(self.n_epochs) +'.png'
        path_lossPlot_save      = os.path.join(path_results_folder, name_loss_file)
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs         = self.n_epochs
        
        # create model and results folder if not already present
        if (not os.path.exists(path_model_folder)):
            os.makedirs(path_model_folder)
        if (not os.path.exists(path_results_folder)):
            os.makedirs(path_results_folder)
        
        # save loss plot
        plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
        
        # save model
        saveModel(self.model, path_model_save)

class DFD_BinClassifier_v2(BinaryClassifier):
    """
        binary classifier for deepfake detection using partial CDDB dataset for the chosen scenario configuration.
        This second version includes usage of ResNet model, with validation set used during training (70% training, 10% validation, 20% testing),
        and data augementation.
        Respect the v1 this version also includes the use of validation set for early stopping.
        
        training model folders:
        
    """
    def __init__(self, scenario, useGPU = True, batch_size = 32, model_type = "resnet_pretrained"):
        """ init classifier

        Args:
            scenario (str): select between "content","group","mix" scenarios:
            - "content", data (real/fake for each model that contains a certain type of images) from a pseudo-category,
            chosen only samples with faces, OOD -> all other data that contains different subject from the one in ID.
            - "group", ID -> assign 2 data groups from CDDB (deep-fake resources, non-deep-fake resources),
            OOD-> the remaining data group (unknown models)
            - "mix", mix ID and ODD without maintaining the integrity of the CDDB groups, i.e take models samples from
            1st ,2nd,3rd groups and do the same for OOD without intersection.
            
            useGPU (bool, optional): flag to use CUDA device or cpu hardware by the model. Defaults to True.
            batch_size (int, optional): batch size used by dataloaders. Defaults is 32.
            model_type (str, optional): choose which model to use, a pretrained ResNet or one from scratch. Defaults is "resnet_pretrained".
        """
        super(DFD_BinClassifier_v2, self).__init__(useGPU = useGPU, batch_size = batch_size, model_type = model_type)
        self.scenario = scenario
            
        # load dataset & dataloader
        self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False, augment= True)
        test_dataset        = CDDB_binary_Partial(scenario = self.scenario, train = False, ood = False, augment= False)
        self.valid_dataset, self.test_dataset = sampleValidSet(trainset= self.train_dataset, testset= test_dataset, useTestSet = True, verbose = True)
        
        # load model
        if self.model_type == "resnet_pretrained":   # fine-tuning
            self.model = ResNet_ImageNet(n_classes = 2).getModel()
            self.path2model_results   = os.path.join(self.path_results, "ImageNet")  # this give a simple name when initialized the module, but training and loading the model, this changes
        else:                                   # training from scratch
            self.model = ResNet(depth_level= 2)
            self.init_weights_kaimingNormal()    # initializaton of the wights
            self.path2model_results   = os.path.join(self.path_results, "RandomIntialization")
            
            
        self.model.to(self.device)
        
        # define loss and final activation function
        self.sigmoid = F.sigmoid
        self.bce     = F.binary_cross_entropy_with_logits
        
        # learning parameters (default)
        self.lr                     = 1e-5
        self.n_epochs               = 40
        self.weight_decay           = 0.001          # L2 regularization term 
        self.patience               = 5              # early stopping patience
        self.early_stopping_trigger = "loss"        # values "acc" or "loss"
        
        self._check_parameters()
     
    def _check_parameters(self):
        if not(self.early_stopping_trigger in ["loss", "acc"]):
            raise ValueError('The early stopping trigger value must be chosen between "loss" and "acc"')
        
     
    def valid(self, epoch, valid_dataloader):
        """
            validation method used mainly for the Early stopping training
        """
        print (f"Validation for the epoch: {epoch} ...")
        
        # set temporary evaluation mode and empty cuda cache
        self.model.eval()
        T.cuda.empty_cache()
        
        # list of losses
        losses = []
        # counters to compute accuracy
        correct_predictions = 0
        num_predictions = 0
        
        for (x,y) in tqdm(valid_dataloader):
            
            x = x.to(self.device)
            # y = y.to(self.device).to(T.float32)
            y = y.to(self.device).to(T.float32)
            
            with T.no_grad():
                with autocast():
                    
                    logits = self.model.forward(x)
                    
                    if self.early_stopping_trigger == "loss":
                        loss = self.bce(input=logits, target=y)   # logits bce version
                        losses.append(loss.item())
                        
                    elif self.early_stopping_trigger == "acc":
                        probs = self.sigmoid(logits)
 
                        # prepare predictions and targets
                        y_pred  = T.argmax(probs, -1).cpu().numpy()  # both are list of int (indices)
                        y       = T.argmax(y, -1).cpu().numpy()
                        
                        # update counters
                        correct_predictions += (y_pred == y).sum()
                        num_predictions += y_pred.shape[0]
                        
        # go back to train mode 
        self.model.train()
        
        if self.early_stopping_trigger == "loss":
            # return the average loss
            loss_valid = sum(losses)/len(losses)
            print(f"Loss from validation: {loss_valid}")
            return loss_valid
        elif self.early_stopping_trigger == "acc":
            # return accuracy
            accuracy_valid = correct_predictions / num_predictions
            print(f"Accuracy from validation: {accuracy_valid}")
            return accuracy_valid
    
    def train(self, name_train):
        
        # define train dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # define valid dataloader
        valid_dataloader = DataLoader(self.valid_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
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
        
        # initialzie the patience counter and history for early stopping
        valid_history       = []
        counter_stopping    = 0
        last_epoch          = 0
        
        # loop over epochs
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # define cumulative loss for the current epoch
            loss_epoch = 0
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
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

                    loss = self.bce(input=logits, target=y)   # logits bce version
                
                # update total loss    
                loss_epoch += loss.item()   # from tensor with single value to int and accumulation
                
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
            
            # include validation here if needed
            criterion = self.valid(epoch=epoch_idx+1, valid_dataloader= valid_dataloader)
            
            # early stopping update
            if self.early_stopping_trigger == "loss":
                valid_history.append(criterion)             
                if epoch_idx > 0:
                    if valid_history[-1] > valid_history[-2]:
                        if counter_stopping >= self.patience:
                            print("Early stop")
                            break
                        else:
                            print("Pantience counter increased")
                            counter_stopping += 1
                    else:
                        print("loss decreased respect previous epoch")
                        
            elif self.early_stopping_trigger == "acc":
                valid_history.append(criterion)
                if epoch_idx > 0:
                    if valid_history[-1] < valid_history[-2]:
                        if counter_stopping >= self.patience:
                            print("Early stop")
                            break
                        else:
                            print("Pantience counter increased")
                            counter_stopping += 1
                    else:
                        print("Accuracy increased respect previous epoch")
                
            # break
        
        
        # create paths and file names for saving training outcomes
        current_date = date.today().strftime("%d-%m-%Y")
        
        path_model_folder       = os.path.join(self.path_models,  name_train + "_" + current_date)
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        path_results_folder     = os.path.join(self.path_results, name_train + "_" + current_date)
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        path_lossPlot_save      = os.path.join(path_results_folder, name_loss_file)
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs        = last_epoch
        
        # create model and results folder if not already present
        if (not os.path.exists(path_model_folder)):
            os.makedirs(path_model_folder)
        if (not os.path.exists(path_results_folder)):
            os.makedirs(path_results_folder)
        
        # save loss plot
        plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
        
        # save model
        saveModel(self.model, path_model_save)
    
    
# [test section] 
if __name__ == "__main__":
    
    # testing functions
    
    def test_workers_dl():                
        dataset = CDDB_binary()
        test_num_workers(dataset, batch_size  =32)   # use n_workers = 8
    
    def binary_classifier_v1():
        bin_classifier = DFD_BinClassifier_v1(useGPU = True, model_type="resnet_pretrained")
        # bin_classifier.train(name_train="resnet50_ImageNet")
        # print(bin_classifier.device)
        bin_classifier.load("resnet50_ImageNet_13-10-2023", 20)
        # bin_classifier.test()
        # bin_classifier.getLayers(show = True)
    
    def test_binary_classifier_v2():
        bin_classifier = DFD_BinClassifier_v2(scenario = "content", useGPU= True, model_type="resnet_pretrained")
        # bin_classifier.load("resnet50_ImageNet_13-10-2023", 20)
        # bin_classifier.getLayers(show = True)
        
        train_dataset  = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment= True)
        test_dataset   = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment= False)
        valid_dataset, test_dataset = sampleValidSet(trainset= train_dataset, testset= test_dataset, useTestSet = True, verbose = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size= 32, num_workers= 8, shuffle= False, pin_memory= True)
        bin_classifier.early_stopping_trigger = "acc"
        print(bin_classifier.valid(epoch=0, valid_dataloader= valid_dataloader))
        
    def train_v2_content_scenario1():
        bin_classifier = DFD_BinClassifier_v2(scenario = "content", useGPU= True, model_type="resnet_pretrained")
        bin_classifier.early_stopping_trigger = "loss"
        bin_classifier.train(name_train="faces_resnet50ImageNet")   #name with the pattern {data scenario}_{model name}, the algorithm include other name decorations
            
        
    train_v2_content_scenario1()

   
    

    
    
    
    
    
    
    
    
    