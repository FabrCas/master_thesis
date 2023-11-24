from    time                                import time
import  random              
from    tqdm                                import tqdm
from    datetime                            import date
import  math

import  torch                               as T
import  numpy                               as np
import  os              

from    torch.nn                            import functional as F
from    torch.optim                         import Adam, lr_scheduler
from    torch.cuda.amp                      import GradScaler, autocast
from    torch.utils.data                    import DataLoader
from    torchvision.transforms              import v2
from    torch.utils.data                    import default_collate

from    utilities                           import plot_loss, saveModel, metrics_binClass, loadModel, test_num_workers, sampleValidSet, \
                                            duration, check_folder, cutmix_image, showImage, image2int, ExpLogger
from    dataset                             import CDDB_binary, CDDB_binary_Partial
from    models                              import ResNet_ImageNet, ResNet, ResNet_EDS, Unet4_Scorer


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
        
        # handle single image, increasing dimensions simulating a batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        elif len(x.shape) <= 2 or len(x.shape) >= 5:
            raise ValueError("The input shape is not compatiple, expected a batch or a single image")
        
        # correct the dtype
        if not (x.dtype is T.float32):
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
    
    @duration    
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
        check_folder(self.path2model_results)
            
        # compute metrics from test data
        metrics_binClass(predictions, targets, predicted_probabilities, epoch_model= str(self.modelEpochs), path_save = self.path2model_results)
    
    def cutmix_custom(self, x, y, prob = 0.5, verbose = False):
        
        uniform_sampled = random .random()
        if  uniform_sampled< prob:
            if verbose: print("Applied CutMix")
            x_cutmix,y_cutmix = cutmix_image(x,y)
            return x_cutmix, y_cutmix
        else:
            return x,y
     
    def reconstruction_loss(self, target, reconstruction, use_abs = True, range255 = False):
        """ 
            reconstruction loss (MSE/MAE) over batch of images
        
        Args:
            - target (T.tensor): image feeded into the netwerk to be reconstructed 
            - reconstruction (T.tensor): image reconstructed by the decoder
            - range255 (boolean): specify with which image range compute the loss
        
        Returns:
            MSE loss (T.Tensor) with one scalar
        
        """
        if range255:
            if use_abs:
                return T.mean(T.abs(image2int(target, True) - image2int(reconstruction, True)))
            else:
                return T.mean(T.square(image2int(target, True) - image2int(reconstruction, True)))
        else:
            if use_abs:
                return T.mean(T.abs(target - reconstruction))
            else:
                return T.mean(T.square(target - reconstruction))
        
    def init_logger(self, path_model):
        """
            path_model -> specific path of the current model training
        """
        
        logger = ExpLogger(path_model=path_model)
        logger.write_config(self._dataConf())
        logger.write_hyper(self._hyperParams())
        try:
            logger.write_model(self.model.getSummary(verbose=False))
        except:
            print("Impossible to retrieve the model structure for logging")
        
        return logger
                
class DFD_BinClassifier_v1(BinaryClassifier):
    """
        binary classifier for deepfake detection using CDDB dataset, first version.
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
        self.version = 1
        
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
        self.model.eval()
        
        # define loss and final activation function
        self.sigmoid = F.sigmoid
        self.bce     = F.binary_cross_entropy_with_logits
        
        # learning hyperparameters (default)
        self.lr             = 1e-5
        self.n_epochs       = 20
        self.weight_decay   = 0.001       # L2 regularization term 
        
        
    def valid(self):
        raise NotImplementedError
    
    @duration
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
        
        # learned epochs by the model initialization
        self.modelEpochs = 0
        
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
        
        path_model_folder       = os.path.join(self.path_models,  name_train + "_v{}_.".format(str(self.version)) + current_date)
        name_model_file         = str(self.n_epochs) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        path_results_folder     = os.path.join(self.path_results, name_train + "_v{}_.".format(str(self.version)) + current_date)
        name_loss_file          = 'loss_'+ str(self.n_epochs) +'.png'
        path_lossPlot_save      = os.path.join(path_results_folder, name_loss_file)
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs         = self.n_epochs
        
        # create model and results folder if not already present
        check_folder(path_model_folder)
        check_folder(path_results_folder)

        # save loss plot
        plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
        plot_loss(loss_epochs, title_plot= name_train, path_save = os.path.join(path_model_folder,name_loss_file), show=False)
        
        # save model
        saveModel(self.model, path_model_save)

class DFD_BinClassifier_v2(BinaryClassifier):
    """
        binary classifier for deepfake detection using partial CDDB dataset for the chosen scenario configuration.
        Model used: ResNet 50 pre-trained on ImageNet and fine-tuned (all layers)
        This second version includes usage of ResNet model, with validation set used during training (70% training, 10% validation, 20% testing),
        and data augementation.
        Respect the v1 this version also includes the use of validation set for early stopping.
        
        training model folders:
        - faces_resnet50_ImageNet_04-11-2023
        - groups_resnet50_ImageNet_05-11-2023
        - mix_resnet50_ImageNet_05-11-2023
        
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
        self.version = 2
        self.scenario = scenario
            
        # load dataset & dataloader
        self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False, augment= True)
        test_dataset        = CDDB_binary_Partial(scenario = self.scenario, train = False, ood = False, augment= False)
        self.valid_dataset, self.test_dataset = sampleValidSet(trainset= self.train_dataset, testset= test_dataset, useOnlyTest = True, verbose = True)
        
        # load model
        if self.model_type == "resnet_pretrained":   # fine-tuning
            self.model = ResNet_ImageNet(n_classes = 2).getModel()
            self.path2model_results   = os.path.join(self.path_results, "ImageNet")  # this give a simple name when initialized the module, but training and loading the model, this changes
        else:                                   # training from scratch
            self.model = ResNet(depth_level= 2)
            self.init_weights_kaimingNormal()    # initializaton of the wights
            self.path2model_results   = os.path.join(self.path_results, "RandomIntialization")
            
        self.model.to(self.device)
        self.model.eval()
        
        # define loss and final activation function
        self.sigmoid = F.sigmoid
        self.bce     = F.binary_cross_entropy_with_logits
        
        # learning hyperparameters (default)
        self.lr                     = 1e-5
        self.n_epochs               = 40
        self.weight_decay           = 0.001          # L2 regularization term 
        self.patience               = 5              # early stopping patience
        self.early_stopping_trigger = "acc"        # values "acc" or "loss"
        
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
    
    @duration
    def train(self, name_train):
        """name_train (str) should include the scenario selected and the model name (i.e. ResNet50), keep this convention {scenario}_{model_name}"""
        
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
        
        # learned epochs by the model initialization
        self.modelEpochs = 0
        
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
        
        path_model_folder       = os.path.join(self.path_models,  name_train + "_v{}_.".format(str(self.version)) + current_date)
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        path_results_folder     = os.path.join(self.path_results, name_train + "_v{}_.".format(str(self.version)) + current_date)
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        path_lossPlot_save      = os.path.join(path_results_folder, name_loss_file)
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs        = last_epoch
        
        # create model and results folder if not already present
        check_folder(path_model_folder)
        check_folder(path_results_folder)
        
        # save loss plot
        plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
        plot_loss(loss_epochs, title_plot= name_train, path_save = os.path.join(path_model_folder,name_loss_file), show=False)
        
        # save model
        saveModel(self.model, path_model_save)

class DFD_BinClassifier_v3(BinaryClassifier):
    """
        binary classifier for deepfake detection using partial CDDB dataset for the chosen scenario configuration.
        Model used: Custom ResNet50 with encoder/decoder structure (ResNet_EDS)
        This third version extends the 2nd version. Including:
        - custom logging functionalities
        - dynamic learning rate using validation set
        - CutMix usage (https://arxiv.org/abs/1905.04899)
        - Simulataneous learning of a decoder module
        - new interpolated loss as alpha * classification loss + beta* reconstruction loss
        changed lr from 1e-5 to 1e-4.
        
        training model folders:
        faces_resnet_50EDS_v3_.17-11-2023
    """
    def __init__(self, scenario, useGPU = True, batch_size = 32):
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
        """
        super(DFD_BinClassifier_v3, self).__init__(useGPU = useGPU, batch_size = batch_size, model_type = "resnet_eds")
        self.version = 3
        self.scenario = scenario
        self.augment_data_train = True
        self.use_cutmix         = True
            
        # load dataset: train, validation and test.
        
        if self.use_cutmix:
            self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False, augment= self.augment_data_train, label_vector= False)  # set label_vector = False for CutMix collate
        else:
             self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False, augment= self.augment_data_train, label_vector= True)
        test_dataset        = CDDB_binary_Partial(scenario = self.scenario, train = False, ood = False, augment= False)
        self.valid_dataset, self.test_dataset = sampleValidSet(trainset= self.train_dataset, testset= test_dataset, useOnlyTest = True, verbose = True)
        
        # load model
        self.model_type = "resnet_eds" 
        self.model = ResNet_EDS(n_channels=3, n_classes=2, use_upsample= False)
          
        self.model.to(self.device)
        self.model.eval()
        
        # define loss and final activation function
        self.sigmoid = F.sigmoid
        self.bce     = F.binary_cross_entropy_with_logits
        
        # learning hyperparameters (default)
        self.lr                     = 1e-4
        self.n_epochs               = 40
        self.weight_decay           = 0.001          # L2 regularization term 
        self.patience               = 5              # early stopping patience
        self.early_stopping_trigger = "acc"        # values "acc" or "loss"
        
        # loss definition + interpolation values for the new loss
        self.loss_name = "bce + reconstruction loss"
        self.alpha_loss             = 0.9  # bce
        self.beta_loss              = 0.1  # reconstruction

        self._check_parameters()
     
    def _check_parameters(self):
        if not(self.early_stopping_trigger in ["loss", "acc"]):
            raise ValueError('The early stopping trigger value must be chosen between "loss" and "acc"')
        if self.alpha_loss + self.beta_loss != 1.: 
            raise  ValueError('Interpolation hyperparams (alpha, beta) should sum up to 1!')
    
    def _hyperParams(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs_max": self.n_epochs,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.patience,
            "early_stopping_trigger": self.early_stopping_trigger
                }
    
    def _dataConf(self):
        
        # load not fixed config specs with try-catch
        try:
            d_out = self.model.decoder_out_fn.__class__.__name__
        except:
            d_out = "empty"
        
        
        return {
            "date_training": date.today().strftime("%d-%m-%Y"),
            "model": self.model_type,
            "decoder_out_activation": d_out,
            "data_scenario": self.scenario,
            "version_train": self.version,
            "optimizer": self.optimizer.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__,
            "loss": self.loss_name,
            "base_augmentation": self.augment_data_train,
            "cutmix": self.use_cutmix,            
            "grad_scaler": True,                # always true
            }
    
    
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
                    
                    encoding = self.model.encoder_module.forward(x)
                    logits   = self.model.scorer_module.forward(encoding)
                    
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
    
    @duration
    def train(self, name_train, test_loop = False):
        """
        Args:
            name_train (str) should include the scenario selected and the model name (i.e. ResNet50), keep this convention {scenario}_{model_name}
        """
        
        # define the model dir path and create the directory
        current_date = date.today().strftime("%d-%m-%Y")    
        path_model_folder       = os.path.join(self.path_models,  name_train + "_v{}_".format(str(self.version)) + current_date)
        check_folder(path_model_folder)
        
        
        # define train dataloader
        train_dataloader = None
        
        if self.use_cutmix:
            # intiialize CutMix (data augmentation/regularization) module and collate function
            
            cutmix = v2.CutMix(num_classes=2)                   # change for non-binary case!
            def collate_cutmix(batch):
                """
                this function apply the CutMix technique with a certain probability (half probability). the batch should be
                defined with idx labels, but cutmix returns a sequence of values (n classes) for each label based on the composition.
                """
                prob = 0.5
                if random.random() < prob:
                    return cutmix(*default_collate(batch))
                else:
                    return default_collate(batch) 
            
            
           
            train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True, collate_fn = collate_cutmix)
        else:
            train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # define valid dataloader
        valid_dataloader = DataLoader(self.valid_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # define the optimization algorithm
        self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        # learning rate scheduler
        if self.early_stopping_trigger == "loss":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr, verbose = True) # reduce of a half the learning rate 
        elif self.early_stopping_trigger == "acc":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr, verbose = True) # reduce of a half the learning rate 
        
        # model in training mode
        self.model.train()
        
        # define the gradient scaler to avoid weigths explosion
        scaler = GradScaler()
        
        # initialize logger
        logger  = self.init_logger(path_model= path_model_folder)
        
        # intialize data structure to keep track of training performance
        loss_epochs = []
        
        # initialzie the patience counter and history for early stopping
        valid_history       = []
        counter_stopping    = 0
        last_epoch          = 0
        
        # learned epochs by the model initialization
        self.modelEpochs = 0
        
        # loop over epochs
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # define cumulative loss for the current epoch and max/min loss
            loss_epoch = 0; max_loss_epoch = 0; min_loss_epoch = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= n_steps):
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # adjust labels if cutmix has been not applied (from indices to one-hot encoding)
                if len(y.shape) == 1:
                    y = T.nn.functional.one_hot(y)
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(True)
                y = y.to(self.device)               # binary int encoding for each sample
                y = y.to(T.float)
                
                # zeroing the gradient
                self.optimizer.zero_grad()
                
                # model forward and loss computation
                with autocast():
                    encoding        = self.model.encoder_module.forward(x)
                    logits          = self.model.scorer_module.forward(encoding)
                    reconstruction  = self.model.decoder_module.forward(encoding)
                    
                    # print(x.shape)
                    # print(reconstruction.shape)
                    # logits = self.model.forward(x)

                    class_loss  = self.bce(input=logits, target=y)   # logits bce version
                    rec_loss    = self.reconstruction_loss(target = x, reconstruction = reconstruction)
                    loss = self.alpha_loss * class_loss + self.beta_loss * rec_loss 
                    
                
                if loss_epoch>max_loss_epoch    : max_loss_epoch = round(loss_epoch,4)
                if loss_epoch<min_loss_epoch    : min_loss_epoch = round(loss_epoch,4)
                
                # update total loss    
                loss_epoch += loss.item()   # from tensor with single value to int and accumulation
                
                # loss backpropagation
                scaler.scale(loss).backward()
                
                # compute updates using optimizer
                scaler.step(self.optimizer)

                # update weights through scaler
                scaler.update()
                
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
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
            
            self.early_stopping_trigger
            # create dictionary with info frome epoch: loss + valid, and log it
            epoch_data = {"epoch": last_epoch, "avg_loss": avg_loss, "max_loss": max_loss_epoch, \
                          "min_loss": min_loss_epoch, self.early_stopping_trigger + "_valid": criterion}
            logger.log(epoch_data)
            
            # test epochs loop for debug   
            if test_loop and last_epoch == 5: break
            
            # lr scheduler step based on validation result
            self.scheduler.step(criterion)
        
        # create path for the model save
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        # create path for the model results
        path_results_folder     = os.path.join(self.path_results, name_train + "_v{}_".format(str(self.version)) + current_date)
        check_folder(path_results_folder)       # create if doesn't exist
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        path_lossPlot_save      = os.path.join(path_results_folder, name_loss_file)
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs        = last_epoch
        
        # save loss plot
        plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
        plot_loss(loss_epochs, title_plot= name_train, path_save = os.path.join(path_model_folder,name_loss_file), show=False)
        
        # save model
        saveModel(self.model, path_model_save)
    
        # terminate the logger
        logger.end_log()

    # Override of superclass forward method
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
        
        # handle single image, increasing dimensions simulating a batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        elif len(x.shape) <= 2 or len(x.shape) >= 5:
            raise ValueError("The input shape is not compatiple, expected a batch or a single image")
        
        # correct the dtype
        if not (x.dtype is T.float32):
            x = x.to(T.float32)
         
        x = x.to(self.device)
        
        
        encoding    = self.model.encoder_module.forward(x) 
        logits      = self.model.scorer_module.forward(encoding)
        probs       = self.sigmoid(logits)
        pred        = T.argmax(probs, -1)
        fake_prob   = probs[:,1]   # positive class probability (fake probability)
        
        return pred, fake_prob, logits

class DFD_BinClassifier_v4(BinaryClassifier):
    """
        binary classifier for deepfake detection using partial CDDB dataset for the chosen scenario configuration.
        Model used: Custom Unet with encoder/decoder structure
        This fourth etends and edits v3,
        - Usage of Unet with scorer model:
        - batch size from 32 to 64
        - early stopping start from half epochs
        
        training model folders:
        - faces_Unet4Scorer_v4_21-11-2023 (bad results)
        -

    """
    def __init__(self, scenario, useGPU = True, batch_size = 64):
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
        """
        super(DFD_BinClassifier_v4, self).__init__(useGPU = useGPU, batch_size = batch_size, model_type = "U-net_Scorer")
        self.version = 4
        self.scenario = scenario
        self.augment_data_train = True
        self.use_cutmix         = True
            
        # load dataset: train, validation and test.
        
        if self.use_cutmix:
            self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False, augment= self.augment_data_train, label_vector= False)  # set label_vector = False for CutMix collate
        else:
             self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False, augment= self.augment_data_train, label_vector= True)
        test_dataset        = CDDB_binary_Partial(scenario = self.scenario, train = False, ood = False, augment= False)
        self.valid_dataset, self.test_dataset = sampleValidSet(trainset= self.train_dataset, testset= test_dataset, useOnlyTest = True, verbose = True)
        
        # load model
        self.model_type = "U-net_Scorer" 
        self.model = Unet4_Scorer(n_channels=3, n_classes=2)
          
        self.model.to(self.device)
        self.model.eval()
        
        # define loss and final activation function
        self.sigmoid = F.sigmoid
        self.bce     = F.binary_cross_entropy_with_logits
        
        # learning hyperparameters (default)
        self.lr                     = 1e-4
        self.n_epochs               = 40
        self.start_early_stopping   = self.n_epochs/2   # epoch to start early stopping
        self.weight_decay           = 0.001             # L2 regularization term 
        self.patience               = 5                 # early stopping patience
        self.early_stopping_trigger = "acc"             # values "acc" or "loss"
        
        # loss definition + interpolation values for the new loss
        self.loss_name = "bce + reconstruction loss"
        self.alpha_loss             = 0.9  # bce
        self.beta_loss              = 0.1  # reconstruction

        self._check_parameters()
     
    def _check_parameters(self):
        if not(self.early_stopping_trigger in ["loss", "acc"]):
            raise ValueError('The early stopping trigger value must be chosen between "loss" and "acc"')
        if self.alpha_loss + self.beta_loss != 1.: 
            raise  ValueError('Interpolation hyperparams (alpha, beta) should sum up to 1!')
    
    def _hyperParams(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs_max": self.n_epochs,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.patience,
            "early_stopping_trigger": self.early_stopping_trigger,
            "early_stopping_start_epoch": self.start_early_stopping
                }
    
    def _dataConf(self):
        
        # load not fixed config specs with try-catch
        try:
            d_out = self.model.decoder_out_fn.__class__.__name__
        except:
            d_out = "empty"
        
        
        return {
            "date_training": date.today().strftime("%d-%m-%Y"),
            "model": self.model_type,
            "decoder_out_activation": d_out,
            "data_scenario": self.scenario,
            "version_train": self.version,
            "optimizer": self.optimizer.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__,
            "loss": self.loss_name,
            "base_augmentation": self.augment_data_train,
            "cutmix": self.use_cutmix,            
            "grad_scaler": True,                # always true
            "features_exp_order": self.model.features_order
            }
    
    
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
                    
                    logits, _, _ = self.model.forward(x) 
                    
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
    
    @duration
    def train(self, name_train, test_loop = False):
        """
        Args:
            name_train (str) should include the scenario selected and the model name (i.e. ResNet50), keep this convention {scenario}_{model_name}
        """
        
        # define the model dir path and create the directory
        current_date = date.today().strftime("%d-%m-%Y")    
        path_model_folder       = os.path.join(self.path_models,  name_train + "_v{}_".format(str(self.version)) + current_date)
        check_folder(path_model_folder)
        
        # define train dataloader
        train_dataloader = None
        
        if self.use_cutmix:
            # intiialize CutMix (data augmentation/regularization) module and collate function
            
            cutmix = v2.CutMix(num_classes=2)                   # change for non-binary case!
            def collate_cutmix(batch):
                """
                this function apply the CutMix technique with a certain probability (half probability). the batch should be
                defined with idx labels, but cutmix returns a sequence of values (n classes) for each label based on the composition.
                """
                prob = 0.5
                if random.random() < prob:
                    return cutmix(*default_collate(batch))
                else:
                    return default_collate(batch) 
            
            
           
            train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True, collate_fn = collate_cutmix)
        else:
            train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # define valid dataloader
        valid_dataloader = DataLoader(self.valid_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # define the optimization algorithm
        self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        # learning rate scheduler
        if self.early_stopping_trigger == "loss":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr, verbose = True) # reduce of a half the learning rate 
        elif self.early_stopping_trigger == "acc":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr, verbose = True) # reduce of a half the learning rate 
        
        # model in training mode
        self.model.train()
        
        # define the gradient scaler to avoid weigths explosion
        scaler = GradScaler()
        
        # initialize logger
        logger  = self.init_logger(path_model= path_model_folder)
        
        # intialize data structure to keep track of training performance
        loss_epochs = []
        
        # initialzie the patience counter and history for early stopping
        valid_history       = []
        counter_stopping    = 0
        last_epoch          = 0
        
        # learned epochs by the model initialization
        self.modelEpochs = 0
        
        # loop over epochs
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # define cumulative loss for the current epoch and max/min loss
            loss_epoch = 0; max_loss_epoch = 0; min_loss_epoch = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= n_steps):
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # adjust labels if cutmix has been not applied (from indices to one-hot encoding)
                if len(y.shape) == 1:
                    y = T.nn.functional.one_hot(y)
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(True)
                y = y.to(self.device)               # binary int encoding for each sample
                y = y.to(T.float)
                
                # zeroing the gradient
                self.optimizer.zero_grad()
                
                # model forward and loss computation
                with autocast():
                    logits, reconstruction, _ = self.model.forward(x) 

                    class_loss  = self.bce(input=logits, target=y)   # logits bce version
                    rec_loss    = self.reconstruction_loss(target = x, reconstruction = reconstruction, use_abs = True)
                    loss = self.alpha_loss * class_loss + self.beta_loss * rec_loss 
                    
                
                if loss_epoch>max_loss_epoch    : max_loss_epoch = round(loss_epoch,4)
                if loss_epoch<min_loss_epoch    : min_loss_epoch = round(loss_epoch,4)
                
                # update total loss    
                loss_epoch += loss.item()   # from tensor with single value to int and accumulation
                
                # loss backpropagation
                scaler.scale(loss).backward()
                
                # compute updates using optimizer
                scaler.step(self.optimizer)

                # update weights through scaler
                scaler.update()
                
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # include validation here if needed
            criterion = self.valid(epoch=epoch_idx+1, valid_dataloader= valid_dataloader)
            
            # initialize not early stopping
            early_exit = False 
            
            # early stopping update
            if self.early_stopping_trigger == "loss" and last_epoch >= self.start_early_stopping:
                valid_history.append(criterion)             
                if epoch_idx > 0:
                    if valid_history[-1] > valid_history[-2]:
                        if counter_stopping >= self.patience:
                            print("Early stop")
                            early_exit = True
                            # break
                        else:
                            print("Pantience counter increased")
                            counter_stopping += 1
                    else:
                        print("loss decreased respect previous epoch")
                        
            elif self.early_stopping_trigger == "acc" and last_epoch >= self.start_early_stopping:
                valid_history.append(criterion)
                if epoch_idx > 0:
                    if valid_history[-1] < valid_history[-2]:
                        if counter_stopping >= self.patience:
                            print("Early stop")
                            early_exit = True
                            # break
                        else:
                            print("Pantience counter increased")
                            counter_stopping += 1
                    else:
                        print("Accuracy increased respect previous epoch")
            
            
            # create dictionary with info frome epoch: loss + valid, and log it
            epoch_data = {"epoch": last_epoch, "avg_loss": avg_loss, "max_loss": max_loss_epoch, \
                          "min_loss": min_loss_epoch, self.early_stopping_trigger + "_valid": criterion}
            logger.log(epoch_data)
            
            # test epochs loop for debug   
            if test_loop and last_epoch == 5: break
            
            # exit for early stopping if is the case
            if early_exit: break 
            
            # lr scheduler step based on validation result
            self.scheduler.step(criterion)
        
        # create path for the model save
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        # create path for the model results
        path_results_folder     = os.path.join(self.path_results, name_train + "_v{}_".format(str(self.version)) + current_date)
        check_folder(path_results_folder)       # create if doesn't exist
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        path_lossPlot_save      = os.path.join(path_results_folder, name_loss_file)
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs        = last_epoch
        
        # save loss plot
        plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
        plot_loss(loss_epochs, title_plot= name_train, path_save = os.path.join(path_model_folder,name_loss_file), show=False)
        
        # save model
        saveModel(self.model, path_model_save)
    
        # terminate the logger
        logger.end_log()

    # Override of superclass forward method
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
        
        # handle single image, increasing dimensions simulating a batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        elif len(x.shape) <= 2 or len(x.shape) >= 5:
            raise ValueError("The input shape is not compatiple, expected a batch or a single image")
        
        # correct the dtype
        if not (x.dtype is T.float32):
            x = x.to(T.float32)
         
        x = x.to(self.device)
        
        logits, _, _ = self.model.forward(x) 
        
        probs       = self.sigmoid(logits)
        pred        = T.argmax(probs, -1)
        fake_prob   = probs[:,1]   # positive class probability (fake probability)
        
        return pred, fake_prob, logits


if __name__ == "__main__":
    #                           [Start test section] 
    def test_workers_dl():                
        dataset = CDDB_binary()
        test_num_workers(dataset, batch_size  =32)   # use n_workers = 8
    
    def test_cutmix1():
        
        cutmix = v2.CutMix(num_classes=2)
   
        def collate_cutmix(batch):
            """
            this function apply the CutMix technique with a certain probability (half probability)
            the batch should be defined with idx labels, but cutmix returns a sequence of values (n classes) for each label
            based on the composition
            """
            prob = 0.5
            if random .random() < prob:
                return cutmix(*default_collate(batch))
            else:
                return default_collate(batch)
        
        dl = DataLoader(CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment= False, label_vector= False), batch_size=8, shuffle= True, collate_fn= collate_cutmix)
        # classifier = BinaryClassifier(useGPU=True, batch_size=8, model_type = "")
        
        show = False
        
        for idx_out, (x,y) in enumerate(dl):
            # x,y torch tensor 
            print(x.shape, y.shape)
            
            # since we are using cutmix and "label_vector"= False in the Dataset parameters, adjust y if is the case
            if len(y.shape) == 1:
                y = T.nn.functional.one_hot(y)
            
            # x_ ,y_ = cutmix(x,y)
           
            for idx in range(len(x)):
                print(y[idx])
                if show :
                    showImage(x[idx])
                
            if idx_out > 5: break

    def test_cutmix2():
        dl = DataLoader(CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment= False, label_vector=True), batch_size=2, shuffle= True)
        classifier = BinaryClassifier(useGPU=True, batch_size=8, model_type = "")
        
        show = True
        
        for idx_out, (x,y) in enumerate(dl):
            # x,y torch tensor 
            # print(x.shape, y.shape)
        
            x_ ,y_ = classifier.cutmix_custom(x.numpy(),y.numpy(), prob= 0.5, verbose= True)
           
            for idx in range(len(x)):
                print("original labels: ",y[idx]," new labels: ", y_[idx])
                if show :
                    showImage(x_[idx])
            print()
                
            if idx_out > 5: break   
            
    # ________________________________ v1  ________________________________

    def test_binary_classifier_v1():
        bin_classifier = DFD_BinClassifier_v1(useGPU = True, model_type="resnet_pretrained")
        print(bin_classifier.device)
        bin_classifier.getLayers(show = True)
        
    def train_v1():
        bin_classifier = DFD_BinClassifier_v1(useGPU = True, model_type="resnet_pretrained")
        bin_classifier.load("resnet50_ImageNet_13-10-2023", 20)
        bin_classifier.train(name_train="resnet50_ImageNet")
        
    def test_v1():
        bin_classifier = DFD_BinClassifier_v1(useGPU = True, model_type="resnet_pretrained")
        bin_classifier.load("resnet50_ImageNet_13-10-2023", 20)
        bin_classifier.test()
    
    # ________________________________ v2  ________________________________

    def test_binary_classifier_v2():
        bin_classifier = DFD_BinClassifier_v2(scenario = "content", useGPU= True, model_type="resnet_pretrained")
        # bin_classifier.load("resnet50_ImageNet_13-10-2023", 20)
        # bin_classifier.getLayers(show = True)
        
        train_dataset  = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment= True)
        test_dataset   = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment= False)
        valid_dataset, test_dataset = sampleValidSet(trainset= train_dataset, testset= test_dataset, useOnlyTest = True, verbose = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size= 32, num_workers= 8, shuffle= False, pin_memory= True)
        print(bin_classifier.valid(epoch=0, valid_dataloader= valid_dataloader))
        
    def train_v2_content_scenario():
        bin_classifier = DFD_BinClassifier_v2(scenario = "content", useGPU= True, model_type="resnet_pretrained")
        bin_classifier.early_stopping_trigger = "acc"
        bin_classifier.train(name_train="faces_resnet_50ImageNet")   #name with the pattern {data scenario}_{model name}, the algorithm include other name decorations
            
    def train_v2_group_scenario():
        bin_classifier = DFD_BinClassifier_v2(scenario = "group", useGPU= True, model_type="resnet_pretrained")
        bin_classifier.early_stopping_trigger = "acc"
        bin_classifier.train(name_train="group_resnet50_ImageNet")   #name with the pattern {data scenario}_{model name}, the algorithm include other name decorations
    
    def train_v2_mix_scenario():
        bin_classifier = DFD_BinClassifier_v2(scenario = "mix", useGPU= True, model_type="resnet_pretrained")
        bin_classifier.early_stopping_trigger = "acc"
        bin_classifier.train(name_train="mix_resnet50_ImageNet")   #name with the pattern {data scenario}_{model name}, the algorithm include other name decorations
    
    def test_v2_metrics(name_model, epoch, scenario):
        bin_classifier = DFD_BinClassifier_v2(scenario = scenario, useGPU= True, model_type="resnet_pretrained")
        bin_classifier.load(name_model, epoch)
        bin_classifier.test()
    
    # ________________________________ v3  ________________________________

    def test_binary_classifier_v3():
        bin_classifier = DFD_BinClassifier_v3(scenario = "content", useGPU= True)
        # bin_classifier.train(name_train= "faces_resnet50EDS")

    def train_v3_content_scenario():
        bin_classifier = DFD_BinClassifier_v3(scenario = "content", useGPU= True)
        bin_classifier.train(name_train= "faces_resnet50EDS")
    
    def test_v3_metrics(name_model, epoch, scenario):
        bin_classifier = DFD_BinClassifier_v3(scenario = scenario, useGPU= True)
        bin_classifier.load(name_model, epoch)
        bin_classifier.test()  
    
    def showReconstruction_v3(name_model, epoch, scenario):
        bin_classifier = DFD_BinClassifier_v3(scenario = scenario, useGPU= True)
        bin_classifier.load(name_model, epoch)
        img, _ = bin_classifier.test_dataset.__getitem__(300)
        print(img.shape)
        showImage(img)
        img     = T.unsqueeze(img, dim= 0).to(bin_classifier.device)
        enc     = bin_classifier.model.encoder_module.forward(img)
        rec_img = bin_classifier.model.decoder_module.forward(enc)
        logits  = bin_classifier.model.scorer_module.forward(enc)
        rec_img = T.squeeze(rec_img, dim = 0)
        print(rec_img)
        print(logits)
        showImage(rec_img)
    
    # ________________________________ v4  ________________________________
    
    def train_v4_content_scenario():
        bin_classifier = DFD_BinClassifier_v4(scenario = "content", useGPU= True)
        bin_classifier.train(name_train= "faces_Unet4Scorer")
    
    def test_v4_metrics(name_model, epoch, scenario):
        bin_classifier = DFD_BinClassifier_v4(scenario = scenario, useGPU= True)
        bin_classifier.load(name_model, epoch)
        bin_classifier.test()
        
    test_v4_metrics(name_model = "faces_Unet4Scorer_v4_21-11-2023", epoch = 12, scenario = "content")
    pass
    #                           [End test section] 

    """ 
            Past test/train launched: 
    
    train_v1()
    test_v1()
    
    train_v2_content_scenario()
    train_v2_group_scenario()
    train_v2_mix_scenario()
    test_v2_metrics(name_model = "faces_resnet50_ImageNet_v2_04-11-2023", epoch = 24 , scenario = "content")
    test_v2_metrics(name_model = "group_resnet50_ImageNet_v2_05-11-2023", epoch = 26 , scenario = "group")
    test_v2_metrics(name_model = "mix_resnet50_ImageNet_v2_05-11-2023", epoch = 21 ,    scenario = "mix")
    
    train_v3_content_scenario()
    test_v3_metrics(name_model = "faces_resnet50EDS_v3_.17-11-2023", epoch = 20, scenario = "content")
    
    train_v4_content_scenario()
    test_v4_metrics(name_model = "faces_Unet4Scorer_v4_21-11-2023", epoch = 12, scenario = "content")
    
    """

    
    
    
    
    
    
    
    
    