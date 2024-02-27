import  os
from    tqdm                                import tqdm
import  numpy                               as np
import  random
import  math
from    sklearn.metrics                     import accuracy_score
from    datetime                            import date

# pytorch
import  torch                               as T
from    torch.nn                            import functional as F
from    torch.optim                         import Adam, lr_scheduler, SGD
from    torch.cuda.amp                      import GradScaler, autocast
from    torch.utils.data                    import DataLoader
from    torch.utils.data                    import random_split
import  copy
from    sklearn.metrics                     import precision_recall_curve, auc, roc_auc_score

# local modules
from    dataset                             import CDDB_binary_Partial, getMNIST_dataset, getCIFAR100_dataset, getCIFAR10_dataset,\
                                                    getFMNIST_dataset, getSVHN_dataset, getDTD_dataset, getTinyImageNet_dataset, OOD_dataset

from    models                              import FC_classifier, get_fc_classifier_Keras, ResNet_EDS, Unet4, ViT_b16_ImageNet, \
                                                    ViT_timm_EA, VAE, Abnormality_module_Encoder_ViT_v3, Abnormality_module_Encoder_ViT_v4
                                                    
from    utilities                           import  duration, saveModel, loadModel, showImage, check_folder, plot_loss, plot_valid, image2int, \
                                                    ExpLogger, include_attention, metrics_multiClass, sampleValidSet, \
                                                    saveJson, metrics_OOD
    

""" 
#####################################################################################################################
                                A simple toy classifier training to test the OOD classification methods
#####################################################################################################################
"""

class MNISTClassifier(object):
    """ simple classifier for the MNIST dataset on handwritten digits, implemented using pytorch """
    def __init__(self, batch_size = 128, useGPU = True):
        super(MNISTClassifier, self).__init__()
        self.useGPU         = useGPU
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        # load train and test data
        self.train_data = getMNIST_dataset(train = True)
        self.test_data  = getMNIST_dataset(train = False)
        
        # load the model
        self.model = FC_classifier(n_channel= 1, n_classes = 10)
        self.model.to(self.device)
        
        # learning hyper-parameters
        self.batch_size     = batch_size
        self.epochs         = 10
        self.lr             = 1e-3
        
        self.cce            = F.cross_entropy    # categorical cross-entropy, takes logits X and sparse labels (no encoding just index) as input
        self.softmax        = F.softmax
        
        # define path
        self.path_test_models   = "./models/test_models"
        self.name_dataset       = "MNIST"
        self.name_model         = "FCClassifier_{}epochs.ckpt".format(self.epochs)
    
    # aux functions
    def init_weights_normal(self):
        print("Initializing weights using Gaussian distribution")
        # Initialize the weights with Gaussian distribution
        for param in self.model.parameters():
            if len(param.shape) > 1:
                T.nn.init.normal_(param, mean=0, std=0.01) 
     
    def load(self):
        path_model = os.path.join(self.path_test_models, self.name_dataset, self.name_model)
        print(f"Loading the model at location: {path_model}")
        try:
            loadModel(self.model, path_model)
            self.model.eval()
        except:
            raise ValueError(f"no model has been found at path {path_model}")
    
    def save(self):
        # create folders if not exists
        if (not os.path.exists(self.path_test_models)):
            os.makedirs(self.path_test_models)
        if (not os.path.exists(os.path.join(self.path_test_models, self.name_dataset))):
            os.makedirs(os.path.join(self.path_test_models, self.name_dataset))
        saveModel(self.model, os.path.join(self.path_test_models, self.name_dataset, self.name_model))
            
    # learning & testing function 
    @duration
    def train(self):
        print(f"Number of samples for the training set: {len(self.train_data)}")
        # init model
        self.init_weights_normal()
        self.model.train()
        
        # get dataloader
        train_dataloader = DataLoader(self.train_data, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # load the optimizer
        optimizer = Adam(self.model.parameters(), lr= self.lr)

        for epoch_idx in range(self.epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # init loss over epoch
            loss_epoch = 0
            
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= len(train_dataloader)):
                
                x = x.to(self.device)
                x.requires_grad_()
                y = y.to(self.device)
                
                # zeroing the gradient  
                optimizer.zero_grad()
                
                # with autocast():
                logits = self.model.forward(x)  
                loss   = self.cce(input=logits, target=y)
                
                # update total loss    
                loss_epoch += loss.item()
                
                # backpropagation and update
                loss.backward()
                optimizer.step()
            
            # compute average loss for the epoch
            avg_loss = loss_epoch/len(train_dataloader)
            print("Average loss: {}".format(avg_loss))
            
        self.test_accuracy()
                    
        # save the model
        self.save()
            
                
    def test_accuracy(self):
        """
            mimimal method which computes accuracy to test perfomance of the model
        """
        print(f"Number of samples for the test set: {len(self.test_data)}")
        
        self.model.eval()
        
        # get dataloader
        test_dataloader = DataLoader(self.test_data, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # define the array to store the result
        predictions = np.empty((0), dtype= np.int16)
        targets     = np.empty((0), dtype= np.int16)
        
        T.cuda.empty_cache()
        for step_idx,(x,y) in tqdm(enumerate(test_dataloader), total= len(test_dataloader)):
            
            x = x.to(self.device)
            y = y.to(self.device).cpu().numpy()
            
            with T.no_grad():
                with autocast():
                    logits  = self.model.forward(x)
                    probs   = self.softmax(input=logits, dim=1)
                    pred    = T.argmax(probs, dim= 1).cpu().numpy()
            
            predictions = np.append(predictions, pred, axis= 0)         
            targets     = np.append(targets,  y, axis  =0)
            
        # compute accuracy
        accuracy = accuracy_score(y_true = targets, y_pred = predictions)
        print(f"Accuracy: {accuracy}")
    
    def forward(self, x):
        if not(isinstance(x, T.Tensor)):
            x = T.tensor(x)
        
        # handle single image, increasing dimensions simulating a batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        elif len(x.shape) <= 2 or len(x.shape) >= 5:
            raise ValueError("The input shape is not compatiple, expected a batch or a single image")
        
        x = x.to(self.device)
        with T.no_grad():
            logits      = self.model.forward(x)
            probs       = self.softmax(logits, dim=1)
            pred        = T.argmax(probs, dim=1)
        
        logits  = logits.cpu().numpy()
        probs   = probs.cpu().numpy()
        pred    = pred.cpu().numpy()
        
        return pred, probs, logits


class MNISTClassifier_keras(object):

    
    def __init__(self, batch_size = 128):
        super(MNISTClassifier_keras, self).__init__()
        self.model = get_fc_classifier_Keras()
        # learning hyper-parameters
        self.batch_size     = batch_size
        self.epochs         = 10
        self.lr             = 1e-3
        
        self.path_test_models   = "./models/test_models"
        self.name_dataset       = "MNIST"
        self.name_model         = "FCClassifierKeras_{}epochs.h5".format(self.epochs)
        
        self._load_mnist()
        
        # tensorflow

        

    def _load_mnist(self):
        from tensorflow                             import keras
        
        mnist = keras.datasets.mnist
        (mnist_train_x, mnist_train_y), (mnist_test_x, mnist_test_y) = mnist.load_data()
        mnist_train_x, mnist_test_x = mnist_train_x/255., mnist_test_x/255.
        self.x_train = mnist_train_x
        self.y_train = mnist_train_y
        self.x_test  = mnist_test_x
        self.y_test  = mnist_test_y


    def train(self):
        from tensorflow                             import keras
        
        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size)

        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print("Training done, test accuracy: {}".format(test_acc))

        # SAVE MODEL
        self.save_model()
        
    def save_model(self):
        """
        Save a Keras model to a file.
        """
        if (not os.path.exists(self.path_test_models)):
                os.makedirs(self.path_test_models)
        if (not os.path.exists(os.path.join(self.path_test_models, self.name_dataset))):
                os.makedirs(os.path.join(self.path_test_models, self.name_dataset))
        path  = os.path.join(self.path_test_models, self.name_dataset, self.name_model)
                
        self.model.save(path)
        print(f"Keras model has beeen saved to {path}")

    def load_model(self):
        """
        Load a Keras model from a file.
        """
        from keras.models                           import load_model
        path  = os.path.join(self.path_test_models, self.name_dataset, self.name_model)
        
        self.model = load_model(path)
        print(f"Keras model has beeen loaded from {path}")

""" 
#####################################################################################################################
                                                Decoding experiments
#####################################################################################################################
"""

class Decoder_ResNetEDS(object):
    """
        class used to learn and test the ability of the resnet decoder to reconstruct the original image
        From v3, removed: valid + early stopping, cutmix augmentation, 
        
        training model folders:
        - faces_resnet50ED_18-11-2023
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
        super(Decoder_ResNetEDS, self).__init__()
        self.useGPU             = useGPU
        self.batch_size         = batch_size
        self.version            = "Experiment test"
        self.scenario           = scenario
        self.augment_data_train = True
        self.use_cutmix         = False
        self.use_upsample       = False
        self.path_models        = "./models/test_models"
        
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
            
        # load dataset: train, validation and test.
        self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False, augment= self.augment_data_train, label_vector= False)  # set label_vector = False for CutMix collate
        
        # load model
        self.model_type = "resnet_eds_decoder" 
        self.model = ResNet_EDS(n_channels=3, n_classes=2, use_upsample= self.use_upsample)
          
        self.model.to(self.device)
        self.model.eval()
        
        # define loss and final activation function
        self.loss_name  = "Reconstruction loss"
        
        # learning hyperparameters (default)
        self.lr                     = 1e-4
        self.n_epochs               = 40
        self.weight_decay           = 0.001          # L2 regularization term 
        
    
    def _hyperParams(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs_max": self.n_epochs,
            "weight_decay": self.weight_decay,
                }
    
    def _dataConf(self):
        
        # load not fixed config specs with try-catch
        try:
            d_out = self.model.decoder_out_fn
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
            "upsample encoding": self.use_upsample
            }
    
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

        
    def load(self, folder_model, epoch):
        try:
            self.path2model         = os.path.join(self.path_models,  folder_model, str(epoch) + ".ckpt")
            self.modelEpochs        = epoch
            loadModel(self.model, self.path2model)
            self.model.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except Exception as e:
            print(e)
            print("No model: {} found for the epoch: {} in the folder: {}".format(folder_model, epoch, self.path_models))
    
    def reconstruction_loss(self, target, reconstruction, range255 = False):
        """ 
            reconstruction loss (MSE) over batch of images
        
        Args:
            - target (T.tensor): image feeded into the netwerk to be reconstructed 
            - reconstruction (T.tensor): image reconstructed by the decoder
            - range255 (boolean): specify with which image range compute the loss
        
        Returns:
            MSE loss (T.Tensor) with one scalar
        
        """
        if range255:
            return T.mean(T.square(image2int(target, True) - image2int(reconstruction, True)))
        else:
            return T.mean(T.square(target - reconstruction))
    
    @duration
    def train(self, name_train, test_loop = False):
        """
        Args:
            name_train (str) should include the scenario selected and the model name (i.e. ResNet50), keep this convention {scenario}_{model_name}
        """
    
        # define the model dir path and create the directory
        current_date = date.today().strftime("%d-%m-%Y")
        path_model_folder       = os.path.join(self.path_models,  name_train + "_" + current_date)
        check_folder(path_model_folder)
        
        # define train dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
                
        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # define the optimization algorithm
        self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        # learning rate scheduler
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        
        # model in training mode
        self.model.train()
        
        # define the gradient scaler to avoid weigths explosion
        scaler = GradScaler()
        
        # initialize logger
        logger  = self.init_logger(path_model= path_model_folder)
        
        # intialize data structure to keep track of training performance
        loss_epochs = []
        
        # initialzie the patience counter and history for early stopping
        last_epoch       = 0
        
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
                    reconstruction  = self.model.decoder_module.forward(encoding)
                
                    loss = self.reconstruction_loss(target = x, reconstruction = reconstruction)
                    
                
                # update total loss    
                loss_epoch += loss.item()   # from tensor with single value to int and accumulation
                
                if loss_epoch>max_loss_epoch    : max_loss_epoch = round(loss_epoch,4)
                if loss_epoch<min_loss_epoch    : min_loss_epoch = round(loss_epoch,4)
                # loss backpropagation
                scaler.scale(loss).backward()
                
                # compute updates using optimizer
                scaler.step(self.optimizer)

                # update weights through scaler
                scaler.update()
                
                # lr scheduler step 
                self.scheduler.step()
                
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # create dictionary with info frome epoch: loss + valid, and log it
            epoch_data = {"epoch": last_epoch, "avg_loss": avg_loss, "max_loss": max_loss_epoch, "min_loss": min_loss_epoch}
            logger.log(epoch_data)
            
            if test_loop and last_epoch == 5: break
        
        # create paths and file names for saving training outcomes        
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        path_lossPlot_save      = os.path.join(path_model_folder, name_loss_file)
        
        
        # save info for the new model trained
        self.modelEpochs        = last_epoch
        
        # save loss plot
        plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
        
        # save model
        saveModel(self.model, path_model_save)
        
        # terminate the logger
        logger.end_log()

class Decoder_Unet(object):
    """
        class used to learn and test the ability of Unet to reconstruct the input image
        
        training model folders:
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
        super(Decoder_Unet, self).__init__()
        self.useGPU             = useGPU
        self.batch_size         = batch_size
        self.version            = "Experiment test"
        self.scenario           = scenario
        self.augment_data_train = True
        self.use_cutmix         = False

        self.path_models        = "./models/test_models"
        
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
            
        # load dataset: train, validation and test.
        self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False, augment= self.augment_data_train, label_vector= False)  # set label_vector = False for CutMix collate
        
        # load model
        self.model_type = "Unet" 
        self.model = Unet4()
        
        self.feature_exp        = self.model.features_order
        
        self.model.to(self.device)
        self.model.eval()
        
        # define loss and final activation function
        self.loss_name  = "Reconstruction loss"
        
        # learning hyperparameters (default)
        self.lr                     = 1e-4
        self.n_epochs               = 40
        self.weight_decay           = 0.001          # L2 regularization term 
        
    
    def _hyperParams(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs_max": self.n_epochs,
            "weight_decay": self.weight_decay,
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
            "features_exp_order": self.feature_exp
            }
    
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

        
    def load(self, folder_model, epoch):
        try:
            self.path2model         = os.path.join(self.path_models,  folder_model, str(epoch) + ".ckpt")
            self.modelEpochs        = epoch
            loadModel(self.model, self.path2model)
            self.model.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except Exception as e:
            print(e)
            print("No model: {} found for the epoch: {} in the folder: {}".format(folder_model, epoch, self.path_models))
    
    def reconstruction_loss(self, target, reconstruction, range255 = False):
        """ 
            reconstruction loss (MSE) over batch of images
        
        Args:
            - target (T.tensor): image feeded into the netwerk to be reconstructed 
            - reconstruction (T.tensor): image reconstructed by the decoder
            - range255 (boolean): specify with which image range compute the loss
        
        Returns:
            MSE loss (T.Tensor) with one scalar
        
        """
        if range255:
            return T.mean(T.square(image2int(target, True) - image2int(reconstruction, True)))
        else:
            return T.mean(T.square(target - reconstruction))
    
    @duration
    def train(self, name_train, test_loop = False):
        """
        Args:
            name_train (str) should include the scenario selected and the model name (i.e. ResNet50), keep this convention {scenario}_{model_name}
        """
    
        # define the model dir path and create the directory
        current_date = date.today().strftime("%d-%m-%Y")
        path_model_folder       = os.path.join(self.path_models,  name_train + "_" + current_date)
        check_folder(path_model_folder)
        
        # define train dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
                
        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # define the optimization algorithm
        self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        # learning rate scheduler
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        
        # model in training mode
        self.model.train()
        
        # define the gradient scaler to avoid weigths explosion
        scaler = GradScaler()
        
        # initialize logger
        logger  = self.init_logger(path_model= path_model_folder)
        
        # intialize data structure to keep track of training performance
        loss_epochs = []
        
        # initialzie the patience counter and history for early stopping
        last_epoch       = 0
        
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
                    # print(x.shape)
                    rec, _ = self.model.forward(x)
                    loss = self.reconstruction_loss(target = x, reconstruction = rec)
                    
                # update total loss    
                loss_epoch += loss.item()   # from tensor with single value to int and accumulation
                
                if loss_epoch>max_loss_epoch    : max_loss_epoch = round(loss_epoch,4)
                if loss_epoch<min_loss_epoch    : min_loss_epoch = round(loss_epoch,4)
                # loss backpropagation
                scaler.scale(loss).backward()
                
                # compute updates using optimizer
                scaler.step(self.optimizer)

                # update weights through scaler
                scaler.update()
                
                # lr scheduler step 
                self.scheduler.step()
                
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # create dictionary with info frome epoch: loss + valid, and log it
            epoch_data = {"epoch": last_epoch, "avg_loss": avg_loss, "max_loss": max_loss_epoch, "min_loss": min_loss_epoch}
            logger.log(epoch_data)
            
            if test_loop and last_epoch == 5: break
        
        # create paths and file names for saving training outcomes        
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        path_lossPlot_save      = os.path.join(path_model_folder, name_loss_file)
        
        # save info for the new model trained
        self.modelEpochs        = last_epoch
        
        # save loss plot
        plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
        
        # save model
        saveModel(self.model, path_model_save)
        
        # terminate the logger
        logger.end_log()

""" 
#####################################################################################################################
                                                ViT Experiment
#####################################################################################################################
"""


class CIFAR_ViT_Classifier(object):
    """ simple classifier for the MNIST dataset on handwritten digits, implemented using pytorch """
    def __init__(self, batch_size = 32, useGPU = True, cifar100 = True):
        super(CIFAR_ViT_Classifier, self).__init__()
        self.useGPU         = useGPU
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        # load train and test data
        if cifar100:
            self.train_data = getCIFAR100_dataset(train = True)
            self.test_data  = getCIFAR100_dataset(train = False)
        else:
            self.train_data = getCIFAR10_dataset(train = True)
            self.test_data  = getCIFAR10_dataset(train = False)
            
        
        # load the model
        # self.model = FC_classifier(n_channel= 1, n_classes = 10)
        
        if cifar100: 
            n_classes = 100
        else:
            n_classes = 10
        
        self.model = ViT_b16_ImageNet(n_classes=n_classes)
        self.model.to(self.device)
        
        # learning hyper-parameters
        self.batch_size     = batch_size
        self.epochs         = 10
        self.lr             = 1e-3
        
        self.cce            = F.cross_entropy    # categorical cross-entropy, takes logits X and sparse labels (no encoding just index) as input
        self.softmax        = F.softmax
        
        # define path
        self.path_test_models   = "./models/test_models"
        if cifar100: 
             self.name_dataset       = "CIFAR100"
        else:
            self.name_dataset       = "CIFAR10"
        self.name_model         = "{}_{}epochs.ckpt".format(self.model.__class__.__name__,self.epochs)
    
    def load(self):
        path_model = os.path.join(self.path_test_models, self.name_dataset, self.name_model)
        print(f"Loading the model at location: {path_model}")
        try:
            loadModel(self.model, path_model)
            self.model.eval()
        except:
            raise ValueError(f"no model has been found at path {path_model}")
    
    def save(self):
        # create folders if not exists
        if (not os.path.exists(self.path_test_models)):
            os.makedirs(self.path_test_models)
        if (not os.path.exists(os.path.join(self.path_test_models, self.name_dataset))):
            os.makedirs(os.path.join(self.path_test_models, self.name_dataset))
        saveModel(self.model, os.path.join(self.path_test_models, self.name_dataset, self.name_model))
            
    # learning & testing function 
    @duration
    def train(self):
        print(f"Number of samples for the training set: {len(self.train_data)}")
        # init model
        self.model.train()
        
        # get dataloader
        train_dataloader = DataLoader(self.train_data, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # load the optimizer
        optimizer = Adam(self.model.parameters(), lr= self.lr)

        for epoch_idx in range(self.epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # init loss over epoch
            loss_epoch = 0
            
            # save model at half epochs
            if (epoch_idx+1) == self.epochs//2:
                self.save()
            
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= len(train_dataloader)):
                
                x = x.to(self.device)
                x.requires_grad_()
                y = y.to(self.device)
                
                # zeroing the gradient  
                optimizer.zero_grad()
                
                # with autocast():
                logits = self.model.forward(x)  
                loss   = self.cce(input=logits, target=y)
                
                # update total loss    
                loss_epoch += loss.item()
                
                # backpropagation and update
                loss.backward()
                optimizer.step()
            
            # compute average loss for the epoch
            avg_loss = loss_epoch/len(train_dataloader)
            print("Average loss: {}".format(avg_loss))
        
    
        # save the model
        self.save()
        
        self.test_accuracy()
                    
                
    def test_accuracy(self):
        """
            mimimal method which computes accuracy to test perfomance of the model
        """
        print(f"Number of samples for the test set: {len(self.test_data)}")
        
        self.model.eval()
        
        # get dataloader
        test_dataloader = DataLoader(self.test_data, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # define the array to store the result
        predictions = np.empty((0), dtype= np.int16)
        targets     = np.empty((0), dtype= np.int16)
        
        T.cuda.empty_cache()
        for step_idx,(x,y) in tqdm(enumerate(test_dataloader), total= len(test_dataloader)):
            
            x = x.to(self.device)
            y = y.to(self.device).cpu().numpy()
            
            with T.no_grad():
                with autocast():
                    logits  = self.model.forward(x)
                    probs   = self.softmax(input=logits, dim=1)
                    pred    = T.argmax(probs, dim= 1).cpu().numpy()
            
            predictions = np.append(predictions, pred, axis= 0)         
            targets     = np.append(targets,  y, axis  =0)
            
        # compute accuracy
        accuracy = accuracy_score(y_true = targets, y_pred = predictions)
        print(f"Accuracy: {accuracy}")
    
    def forward(self, x):
        if not(isinstance(x, T.Tensor)):
            x = T.tensor(x)
        
        # handle single image, increasing dimensions simulating a batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        elif len(x.shape) <= 2 or len(x.shape) >= 5:
            raise ValueError("The input shape is not compatiple, expected a batch or a single image")
        
        x = x.to(self.device)
        
        with T.no_grad():
            logits = self.model.forward(x)
            probs       = self.softmax(logits, dim=1)
            pred        = T.argmax(probs, dim=1)
        
        logits  = logits.cpu().numpy()
        probs   = probs.cpu().numpy()
        pred    = pred.cpu().numpy()
        
        return pred, probs, logits

class CIFAR_ViTEA_Classifier(object):
    """ simple classifier for the MNIST dataset on handwritten digits, implemented using pytorch """
    def __init__(self, prog_model = 2, batch_size = 64, useGPU = True, cifar100 = True):
        super(CIFAR_ViTEA_Classifier, self).__init__()
        self.useGPU         = useGPU
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        # load train and test data
        if cifar100:
            self.train_data = getCIFAR100_dataset(train = True)
            self.test_data  = getCIFAR100_dataset(train = False)
        else:
            self.train_data = getCIFAR10_dataset(train = True)
            self.test_data  = getCIFAR10_dataset(train = False)
            
        
        # load the model
        # self.model = FC_classifier(n_channel= 1, n_classes = 10)
        
        if cifar100: 
            n_classes = 100
        else:
            n_classes = 10
            
        self.model = ViT_timm_EA(n_classes=n_classes, prog_model=prog_model) 
        self.model.to(self.device)
        
        # learning hyper-parameters
        self.batch_size     = batch_size
        self.epochs         = 50
        self.lr             = 1e-3
        
        self.cce            = F.cross_entropy    # categorical cross-entropy, takes logits X and sparse labels (no encoding just index) as input
        self.softmax        = F.softmax
        
        # define path
        self.path_test_models   = "./models/test_models"
        if cifar100: 
             self.name_dataset       = "CIFAR100"
        else:
            self.name_dataset       = "CIFAR10"
        self.name_model         = "{}_{}epochs.ckpt".format(self.model.__class__.__name__,self.epochs)
    
    def load(self):
        path_model = os.path.join(self.path_test_models, self.name_dataset, self.name_model)
        print(f"Loading the model at location: {path_model}")
        try:
            loadModel(self.model, path_model)
            self.model.eval()
        except:
            raise ValueError(f"no model has been found at path {path_model}")
    
    def save(self):
        # create folders if not exists
        if (not os.path.exists(self.path_test_models)):
            os.makedirs(self.path_test_models)
        if (not os.path.exists(os.path.join(self.path_test_models, self.name_dataset))):
            os.makedirs(os.path.join(self.path_test_models, self.name_dataset))
        saveModel(self.model, os.path.join(self.path_test_models, self.name_dataset, self.name_model))
            
    # learning & testing function 
    @duration
    def train(self):
        print(f"Number of samples for the training set: {len(self.train_data)}")
        # init model
        self.model.train()
        
        # get dataloader
        train_dataloader = DataLoader(self.train_data, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # load the optimizer
        optimizer = Adam(self.model.parameters(), lr= self.lr)
        
        for epoch_idx in range(self.epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # init loss over epoch
            loss_epoch = 0
            
            # save model at half epochs
            if (epoch_idx+1) == self.epochs//2:
                self.save()
                
            max_value_att_map = []
            
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= len(train_dataloader)):
                
                # if step_idx == 2: break
            
                x = x.to(self.device)
                # x = self.trans(x)
                x.requires_grad_(True)
                y = y.to(self.device)
                
                # zeroing the gradient  
                optimizer.zero_grad()
                
                # with autocast():
                logits, _ , att_map = self.model.forward(x)  
                loss   = self.cce(input=logits, target=y)
                
                
                # store max value att_map
                
                max_value_att_map.append(T.max(att_map).detach().cpu().item())
                
                # update total loss    
                loss_epoch += loss.item()
                
                # backpropagation and update
                loss.backward()
                optimizer.step()
            
            # compute average loss for the epoch
            avg_loss = loss_epoch/len(train_dataloader)
            print("Average loss: {}".format(avg_loss))
            
            # compute averege max value in attention
            avg_max_attention = sum(max_value_att_map)/len(max_value_att_map)
            print("Average max attention value: {}".format(avg_max_attention))

            
            # if epoch_idx == 2: break
    
        # save the model
        self.save()
        
        self.test_accuracy()
                    
                
    def test_accuracy(self):
        """
            mimimal method which computes accuracy to test perfomance of the model
        """
        print(f"Number of samples for the test set: {len(self.test_data)}")
        
        self.model.eval()
        
        # get dataloader
        test_dataloader = DataLoader(self.test_data, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # define the array to store the result
        predictions = np.empty((0), dtype= np.int16)
        targets     = np.empty((0), dtype= np.int16)
        
        T.cuda.empty_cache()
        for step_idx,(x,y) in tqdm(enumerate(test_dataloader), total= len(test_dataloader)):
            
            x = x.to(self.device)
            # x = self.trans(x)
            y = y.to(self.device).cpu().numpy()
            
            with T.no_grad():
                with autocast():
                    logits, _, _  = self.model.forward(x)
                    probs   = self.softmax(input=logits, dim=1)
                    pred    = T.argmax(probs, dim= 1).cpu().numpy()
            
            predictions = np.append(predictions, pred, axis= 0)         
            targets     = np.append(targets,  y, axis  =0)
            
        # compute accuracy
        accuracy = accuracy_score(y_true = targets, y_pred = predictions)
        print(f"Accuracy: {accuracy}")
    
    def forward(self, x):
        """ network forward

        Args:
            x (T.Tensor): input image/images

        Returns:
            pred: label: 0 -> real, 1 -> fake
        """
        self.model.eval()
        
        
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
        
        logits, _, _  = self.model.forward(x) 
        
        probs       = self.sigmoid(logits)    # change to softmax in multi-class context
        pred        = T.argmax(probs, -1)
        fake_prob   = probs[:,1]   # positive class probability (fake probability)
        
        return pred, fake_prob, logits



""" 
#####################################################################################################################
                                                OOD benchmark CIFAR 10/100
#####################################################################################################################
"""
# Model training

# code inspired from bin_ViTClassifier module and adapted
class CIFAR_VITEA_benchmark(object):
    """ simple classifier for the MNIST dataset on handwritten digits, implemented using pytorch """
    def __init__(self, prog_model = 3, batch_size = 64, useGPU = True, cifar100 = False, image_size = 224):
        super(CIFAR_VITEA_benchmark, self).__init__()
        self.useGPU         = useGPU
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        self.image_size = image_size
        only_transfNorm = True
        
        augmentation = True # train augmentation
        
        # load train and test data
        if cifar100:
            self.train_data = getCIFAR100_dataset(train = True, augment = augmentation, resolution= self.image_size)
            self.test_data  = getCIFAR100_dataset(train = False, augment = False,  resolution= self.image_size)
        else:
            self.train_data = getCIFAR10_dataset(train = True, augment = augmentation,  resolution= self.image_size)
            self.test_data  = getCIFAR10_dataset(train = False, augment = False,  resolution= self.image_size)
            
        
        # load the model
        
        if cifar100: 
            n_classes = 100
        else:
            n_classes = 10
            
        self.model = ViT_timm_EA(n_classes=n_classes, prog_model=prog_model ,only_transfNorm= only_transfNorm) 
        self.model.to(self.device)
        self.autoencoder = VAE(image_size= image_size) 
        self.autoencoder.to(self.device)
        
        self.path_models    = "./models/benchmarks"
        self.path_results   = "./results/benchmarks"
        
        
        # learning hyper-parameters
        self.batch_size     = batch_size
        self.epochs         = 50
        self.lr             = 1e-3# 1e-3
        self.lr_ae          = 1e-3
        self.weight_decay   = 1e-3
        
        # self.cce            = F.cross_entropy    # categorical cross-entropy, takes logits X and sparse labels (no encoding just index) as input
        self.cce            = T.nn.CrossEntropyLoss()
        self.softmax        = F.softmax
        self.mse            = T.nn.MSELoss()
        self.mae            = T.nn.L1Loss()
        
        # define path
        if cifar100: 
             self.name_dataset       = "cifar100"
        else:
            self.name_dataset       = "cifar10"
            
        # self.name_model         = "{}_{}epochs.ckpt".format(self.model.__class__.__name__,self.epochs)
    
    def load_classifier(self, name_folder, epoch):
        name_model = "{}.ckpt".format(epoch)
        path_model = os.path.join(self.path_models, self.name_dataset, name_folder, name_model)
        
        self.train_name = name_folder
        
        self.path2classifier            = os.path.join(self.path_models, self.name_dataset, name_folder)
        self.path2results_classifier    = os.path.join(self.path_results, self.name_dataset, name_folder)
        
        print(f"Loading the model at location: {path_model}")
        try:
            loadModel(self.model, path_model)
            self.model.eval()
        except:
            raise ValueError(f"no model has been found at path {path_model}")
        
    def load_autoencoder(self, name_folder, epoch):
        name_model = "AE_{}.ckpt".format(epoch)
        path_model = os.path.join(self.path_models, self.name_dataset, name_folder, name_model)
        
        self.train_name = name_folder
        
        print(f"Loading the model at location: {path_model}")
        try:
            loadModel(self.model, path_model)
            self.model.eval()
        except:
            raise ValueError(f"no model has been found at path {path_model}")
    
    def load(self, name_folder, epoch_classifier, epoch_ae):
        self.load_classifier(name_folder, epoch_classifier)
        self.load_autoencoder(name_folder, epoch_ae)
    
    def save_classifier(self, name_folder, name_model, model_dict = None):
        # create folders if not exists
        if (not os.path.exists(self.path_models)):
            os.makedirs(self.path_models)
        if (not os.path.exists(os.path.join(self.path_models, self.name_dataset, name_folder))):
            os.makedirs(os.path.join(self.path_models, self.name_dataset, name_folder))
        
        if model_dict is None:
            saveModel(self.model, os.path.join(self.path_models, self.name_dataset, name_folder, name_model))
        else:
            saveModel(model_dict, os.path.join(self.path_models, self.name_dataset, name_folder, name_model), is_dict=True)
         
    def save_autoencoder(self, name_folder, name_model, model_dict = None):
        # create folders if not exists
        if (not os.path.exists(self.path_models)):
            os.makedirs(self.path_models)
        if (not os.path.exists(os.path.join(self.path_models, self.name_dataset, name_folder))):
            os.makedirs(os.path.join(self.path_models, self.name_dataset, name_folder))
        
        if model_dict is None:
            saveModel(self.autoencoder, os.path.join(self.path_models, self.name_dataset, name_folder, name_model))
        else:
            saveModel(model_dict, os.path.join(self.path_models, self.name_dataset, name_folder, name_model), is_dict=True) 
           
    # learning & testing function 
    
    # classifier 
    def valid_classifier(self, valid_dataloader): 
        """
            validation method used mainly for the Early stopping training
        """
        print (f"Validation ...")
        
        # set temporary evaluation mode and empty cuda cache
        self.model.eval()
        
        T.cuda.empty_cache()
        
        # counters to compute accuracy
        correct_predictions = 0
        num_predictions = 0
        
        for (x,y) in tqdm(valid_dataloader):
            x = x.to(self.device)

            with T.no_grad():
                out     = self.model.forward(x) 
                logits  = out[0]
                

                # prepare predictions and targets
                pred    = self.softmax(logits, dim=1)
                y_pred  = T.argmax(pred, -1).cpu().numpy()  # both are list of int (indices)
                # y       = T.argmax(y, -1).cpu().numpy()
                y       = y.cpu().numpy() 
                
                # update counters
                correct_predictions += (y_pred == y).sum()
                num_predictions += y_pred.shape[0]
            
        # go back to train mode 
        self.model.train()
    
        # return accuracy
        accuracy_valid = correct_predictions / num_predictions
        print(f"Accuracy from validation: {accuracy_valid}")
        return accuracy_valid
    
    @duration
    def train_classifier(self, add_in_name="", start_epoch = 0, end_epoch = None):
        
        # check args validity
        if start_epoch != 0:
            if end_epoch is None:
                raise ValueError("End epoch is not specified (required if start epoch is not 0)")
            if (end_epoch - start_epoch) <= 0:
                raise ValueError("end epoch should be grater than start epoch")
        
        if not(end_epoch is None):
            self.epochs = end_epoch

        # define the train name, if already trained just obtain from class attribute
        if start_epoch == 0:
            current_date = date.today().strftime("%d-%m-%Y")    
            name_folder = "train_" + add_in_name + "_" + current_date
            self.train_name = name_folder
        else:
            print(f"Starting training from epoch n° {start_epoch}")
            name_folder = self.train_name
        
        self.path2classifier            = os.path.join(self.path_models, self.name_dataset, name_folder)
        self.path2results_classifier    = os.path.join(self.path_results, self.name_dataset, name_folder)
        
        print(f"Number of samples for the training set: {len(self.train_data)}")
        # init model
        self.model.train()
        
        # split train in train and valid
        test_size = int(0.8 * len(self.test_data))
        val_size = len(self.test_data) - test_size
        _, val_dataset = random_split(self.test_data, [test_size, val_size])
  
        # get dataloader
        train_dataloader = DataLoader(self.train_data, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        valid_dataloader = DataLoader(val_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)

        # load the optimizer
        # optimizer = Adam(self.model.parameters(), lr= self.lr)
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum= 0.9)
        
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[math.ceil(self.epochs*0.25), math.ceil(self.epochs*0.5), math.ceil(self.epochs* 0.75)], gamma=0.5)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0 = 5, T_mult =2, eta_min=1e-6)
        
        # scaler = GradScaler()
        
        if start_epoch != 0:
            loss_epochs     = [*[None] * start_epoch]
            valid_history   = [*[None] * start_epoch]
        else:
            loss_epochs     = []
            valid_history   = []
        
        # define best validation results
        best_valid = 0                       # to maximize
        best_valid_epoch = start_epoch
        # initialize ditctionary best models
        best_model_dict = copy.deepcopy(self.model.state_dict())
        
        for epoch_idx in range((self.epochs - start_epoch)):
            
            # include the start epoch in the idx
            epoch_idx = epoch_idx + start_epoch
            
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # init loss over epoch
            loss_epoch = 0
            
            max_value_att_map = []
            
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= len(train_dataloader)):
                
                
                # showImage(x[0])
                
                x = x.to(self.device)
                # x = self.trans(x)
                x.requires_grad_(True)
                y = y.to(self.device)
                
                # zeroing the gradient  
                optimizer.zero_grad()
                
                # with autocast():
                out = self.model.forward(x)  
                logits = out[0]
                att_map = out[2]
                
                loss   = self.cce(input=logits, target=y)
                
                # store max value att_map
                max_value_att_map.append(T.max(att_map).detach().cpu().item())
                
                # update total loss    
                loss_epoch += loss.item()
                
                # backpropagation and update
                
                # scaler.scale(loss).backward()
                loss.backward()
                
                # scaler.step(optimizer=optimizer)
                optimizer.step()
                
                # scaler.update()
                
                scheduler.step()
            
            # compute average loss for the epoch
            avg_loss = loss_epoch/len(train_dataloader)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # compute averege max value in attention
            avg_max_attention = sum(max_value_att_map)/len(max_value_att_map)
            print("Average max attention value: {}".format(avg_max_attention))
            
            # validation
            criterion = self.valid_classifier(valid_dataloader= valid_dataloader)
            valid_history.append(criterion)
            # look for best models *.*


            if criterion > best_valid:
                best_valid = criterion
                best_model_dict = copy.deepcopy(self.model.state_dict())
                best_valid_epoch = epoch_idx+1
                print("** new best classifier **")
                
            # test accuracy
            accuracy_test = self.test_accuracy()  
            
            # if accuracy_test >= 95:
            #     break

            
            # if epoch_idx == 2: break
        check_folder(os.path.join(self.path_models, self.name_dataset, name_folder))
        check_folder(os.path.join(self.path_results, self.name_dataset, name_folder))
        
        plot_loss(loss_epochs, title_plot= "classifier", path_save = os.path.join(self.path_models, self.name_dataset, name_folder, 'loss_'+ str(self.epochs) +'.png'))
        plot_loss(loss_epochs, title_plot= "classifier", path_save = os.path.join(self.path_results, self.name_dataset, name_folder, 'loss_'+ str(self.epochs) +'.png'), show= False)
        plot_valid(valid_history, title_plot= "classifier accuracy", path_save = os.path.join(self.path_models, self.name_dataset, name_folder, 'valid_'+ str(self.epochs) +'.png'))
        plot_valid(valid_history, title_plot= "classifier accuracy", path_save = os.path.join(self.path_results, self.name_dataset, name_folder, 'valid_'+ str(self.epochs) +'.png'), show= False)
        
        # save the model
        name_model      = "{}.ckpt".format(str(self.epochs))
        name_best_model = "{}.ckpt".format(str(best_valid_epoch))
        
        self.save_classifier(name_folder, name_model)
        self.save_classifier(model_dict = best_model_dict, name_folder=name_folder, name_model=name_best_model)
        # self.test_accuracy()
                    
    def test_accuracy(self):
        """
            mimimal method which computes accuracy to test perfomance of the model
        """
        print(f"Number of samples for the test set: {len(self.test_data)}")
        
        self.model.eval()
        
        # get dataloader
        test_dataloader = DataLoader(self.test_data, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # define the array to store the result
        predictions = np.empty((0), dtype= np.int16)
        targets     = np.empty((0), dtype= np.int16)
        
        T.cuda.empty_cache()
        for step_idx,(x,y) in tqdm(enumerate(test_dataloader), total= len(test_dataloader)):
            
            x = x.to(self.device)
            # x = self.trans(x)
            y = y.to(self.device).cpu().numpy()
            
            with T.no_grad():
                # with autocast():
                out  = self.model.forward(x)
                logits = out[0]
                probs   = self.softmax(input=logits, dim=1)
                pred    = T.argmax(probs, dim= 1).cpu().numpy()
            
            predictions = np.append(predictions, pred, axis= 0)         
            targets     = np.append(targets,  y, axis  =0)
            
        # compute accuracy
        accuracy = accuracy_score(y_true = targets, y_pred = predictions)
        print(f"Accuracy: {accuracy}")
        
        return accuracy
        
    @duration
    def test(self, name_folder, epoch):
        """
            Void function that computes the binary classification metrics

            Params:
            load_data (boolean): parameter used to specify if is required to load data or has been already done, Default is True
        """
        
        self.load_classifier(epoch=epoch, name_folder= name_folder)
        
        
        # define test dataloader
        test_dataloader = DataLoader(self.test_data, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)

        # compute number of batches for test
        n_steps = len(test_dataloader)
        print("Number of batches for test: {}".format(n_steps))
        
        # model in evaluation mode
        self.model.eval()  
        
        # define the array to store the result
        predictions             = np.empty((0), dtype= np.int16)
        targets                 = np.empty((0), dtype= np.int16)
        
        T.cuda.empty_cache()
        
        # loop over steps
        for step_idx,(x,y) in tqdm(enumerate(test_dataloader), total= n_steps):

            x = x.to(self.device)
            # y = y.to(self.device)               # y -> int index for each sample 

            
            with T.no_grad():
                # with autocast():
                pred, _, _ = self.forward(x)
                pred = pred.cpu().numpy().astype(int)

            
            predictions             = np.append(predictions, pred, axis  =0)
            targets                 = np.append(targets,  y, axis  =0)
      
        # create folder for the results
        path_save_results = os.path.join(self.path_results, self.name_dataset, name_folder)
        
        check_folder(path_save_results)
            
        # compute metrics from test data
        metrics_multiClass(predictions, targets, epoch_model= str(epoch), path_save = path_save_results,\
                           labels_indices= list(range(len(self.train_data.classes))), labels_name= self.train_data.classes)
    
    def forward(self, x):
        """ network forward

        Args:
            x (T.Tensor): input image/images

        Returns:
            pred: label: 0 -> real, 1 -> fake
        """
        self.model.eval()
        
        
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
        
        out  = self.model.forward(x) 
        logits = out[0]
        
        probs       = self.softmax(logits, dim=1)    # change to softmax in multi-class context
        pred        = T.argmax(probs, -1)
        fake_prob   = probs[:,1]   # positive class probability (fake probability)
        
        return pred, fake_prob, logits
    
    # autoencoder
    def valid_autoencoder(self, valid_dataloader):
        print (f"Validation...")
        
        # set temporary evaluation mode and empty cuda cache
        self.model.eval()
        self.autoencoder.eval()
        T.cuda.empty_cache()
        
        # list of losses

        losses_ae   = []
        
        for (x,_) in tqdm(valid_dataloader):
            
            x = x.to(self.device)

            
            with T.no_grad():
                out = self.model.forward(x) 
                att_maps    = out[2]
              
                # Autoencoder criterion
                x_ae = att_maps.clone().to(device =self.device)
                
                # if self.model_ae_type == "vae":
                #     rec_att_map, _, _ = self.autoencoder(x_ae)
                # else:
                rec_att_map = self.autoencoder.forward(x_ae)
                loss_ae     = self.mae(rec_att_map, x_ae)
                
                losses_ae.append(loss_ae.cpu().item())
                        
        # go back to train mode 
        self.autoencoder.train()
        
        loss_ae_valid = sum(losses_ae)/len(losses_ae)
        print(f"Loss from validation: {loss_ae_valid}")
        return loss_ae_valid
    
    def train_ae(self, name_folder):
        
        self.autoencoder.train()
        
        # split train in train and valid
        
        train_size = int(0.9 * len(self.train_data))
        val_size = len(self.train_data) - train_size
        train_dataset, val_dataset = random_split(self.train_data, [train_size, val_size])
  
        # get dataloader
        train_dataloader = DataLoader(train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        valid_dataloader = DataLoader(val_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)

        # load the optimizer
        optimizer = Adam(self.autoencoder.parameters(), lr= self.lr_ae, weight_decay= self.weight_decay)
        
        loss_epochs     = []
        valid_history   = []
        
        # define best validation results
        best_valid = math.inf                       # to minimize
        best_valid_epoch = 0
        # initialize ditctionary best models
        best_model_dict = copy.deepcopy(self.autoencoder.state_dict())
        
        for epoch_idx in range(self.epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # init loss over epoch
            loss_epoch = 0
            
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= len(train_dataloader)):

            
                x = x.to(self.device)
                x.requires_grad_(False)

                # model forward and loss computation
                with T.no_grad():
                    
                    out = self.model.forward(x) 
                    att_maps    = out[2]
                    
                # zeroing the gradient  
                optimizer.zero_grad()
                
                x_ae = att_maps.detach().clone()
                x_ae.requires_grad_(True)
                x_ae.to(device=self.device)
                
                rec_att_map, mean, logvar = self.autoencoder.forward(x_ae, train=True)
                loss     = self.autoencoder.loss_function(rec_att_map, x_ae, mean, logvar, use_bce=False)
                
                # with autocast():
                #     out = self.model.forward(x)  
                #     logits = out[0]
                #     att_map = out[2]
                    
                #     loss   = self.cce(input=logits, target=y)
                
                # update total loss    
                loss_epoch += loss.item()
                
                # backpropagation and update
                loss.backward()
                optimizer.step()
            
            # compute average loss for the epoch
            avg_loss = loss_epoch/len(train_dataloader)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # validation
            criterion = self.valid_autoencoder(valid_dataloader= valid_dataloader)
            valid_history.append(criterion)
            # look for best models *.*


            if criterion < best_valid:
                best_valid = criterion
                best_model_dict = copy.deepcopy(self.model.state_dict())
                best_valid_epoch = epoch_idx+1
                print("** new best autoencoder **")   

            
            # if epoch_idx == 2: break
        check_folder(os.path.join(self.path_models, self.name_dataset, name_folder))
        check_folder(os.path.join(self.path_results, self.name_dataset, name_folder))
        
        plot_loss(loss_epochs, title_plot= "AE", path_save = os.path.join(self.path_models, self.name_dataset, name_folder, 'loss_ae_'+ str(self.epochs) +'.png'))
        plot_loss(loss_epochs, title_plot= "AE", path_save = os.path.join(self.path_results, self.name_dataset, name_folder, 'loss_ae_'+ str(self.epochs) +'.png'), show= False)
        plot_valid(valid_history, title_plot= "AE loss", path_save = os.path.join(self.path_models, self.name_dataset, name_folder, 'valid_ae_'+ str(self.epochs) +'.png'))
        plot_valid(valid_history, title_plot= "AE loss", path_save = os.path.join(self.path_results, self.name_dataset, name_folder, 'valid_ae_'+ str(self.epochs) +'.png'), show= False)
        
        # save the model
        name_model      = "AE_{}.ckpt".format(str(self.epochs))
        name_best_model = "AE_{}.ckpt".format(str(best_valid_epoch))
        
        self.save_autoencoder(name_folder = name_folder, name_model = name_model)
        self.save_autoencoder(model_dict = best_model_dict, name_folder=name_folder, name_model=name_best_model)

# code inspired from ood_detection module and adapted 
class OOD_Classifier(object):
    
    def __init__(self, id_data_test = None, ood_data_test = None, id_data_train = None, ood_data_train = None, useGPU = True):
        super(OOD_Classifier, self).__init__()
        
        
        # train data
        self.id_data_train  = id_data_train
        self.ood_data_train = ood_data_train
        # test sets
        self.id_data_test   = id_data_test
        self.ood_data_test  = ood_data_test
        
        # classifier types
        self.types_classifier = ["bin_class", "multi_class", "multi_label_class"]
        
        # general paths
        self.path_models    = "./models/ood_detection"
        self.path_results   = "./results/ood_detection"
        
        # execution specs
        self.useGPU = useGPU
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        
        # hyper-params
        self.batch_size     =  256  # 32 -> basic, 64/128 -> encoder
        
    #                                      data aux functions
    def compute_class_weights(self, verbose = False, positive = "ood", multiplier = 1, normalize = True, only_positive_weight = False):
        
        # TODO look https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        # to implement just pos weight computation 
        
        
        """ positive (str), if ood is used 1 to represent ood labels and 0 for id. The opposite behavior is obtained using "id" """
        
        print("\n\t\t[Computing class weights for the training set]\n")
        
        # set modality to load just the label
        self.dataset_train.set_only_labels(True)
        loader = DataLoader(self.dataset_train, batch_size= None,  num_workers = 8)  # self.dataset_train is instance of OOD_dataset class

        
        # compute occurrences of labels
        class_freq={}
        total = len(self.dataset_train)
        self.samples_train = total

    
        for y in tqdm(loader, total = len(loader)):
            
            y = y.detach().cpu().tolist()
                
            if positive == "ood":
                l = y[1]
            elif positive == "id":
                l = y[0]
            else:
                ValueError('invalid value for the parameter "positive", choose between "ood  and "id"')
            
            if l not in class_freq.keys():
                class_freq[l] = 1
            else:
                class_freq[l] = class_freq[l]+1
        if verbose: print("class_freq -> ", class_freq)
        
        # compute the weights   
        class_weights = []
        for class_ in sorted(class_freq.keys()):
            freq = class_freq[class_]
            class_weights.append(round(total/freq,5))

        # normalize class weights with values between 0 and 1
        if normalize:
            max_value = max(class_weights)
            class_weights = [item/max_value for item in class_weights]
            
        # proportional increase over the weights
        if multiplier != 1:
            class_weights = [item * multiplier for item in class_weights]
        
        print("Class_weights-> ", class_weights)
                
        # turn back in loading modality sample + label
        self.dataset_train.set_only_labels(False)
        
        if only_positive_weight:
            print("Computing weight only for the positive class (label 1)")
            pos_weight = class_weights[1]/class_weights[0]
            print("positive class weight-> ", pos_weight)
            
            return [pos_weight]
        else:
            return class_weights 
    
    def compute_metrics_ood(self, id_data, ood_data, path_save = None, positive_reversed = False, epoch = None):
            """_
                aux function used to compute fpr95, detection error and relative threshold.
                con be selected the positive label, the defulat one is for ID data, set to False
                with the parameter "positive_reversed" for abnormality detection (OOD).
            """
            target = np.zeros((id_data.shape[0] + ood_data.shape[0]), dtype= np.int32)
            
            if positive_reversed:
                target[id_data.shape[0]:] += 1
            else:
                target[:id_data.shape[0]] += 1
                
            # print(target)
            
            predictions = np.squeeze(np.vstack((id_data, ood_data)))
            # get metrics and save AUROC plot
            metrics_ood = metrics_OOD(targets=target, pred_probs= predictions, pos_label= 1, path_save_plot = path_save, epoch = epoch)
            
            return metrics_ood 

    def compute_aupr(self, labels, pred):        
        p, r, _ = precision_recall_curve(labels, pred)
        return  auc(r, p)
    
    def compute_auroc(self, labels, pred):
        return roc_auc_score(labels, pred)
        
    def compute_curves(self, id_data, ood_data, positive_reversed = False):
        """_
            compute AUROC and AUPR, defining the vector of labels from the length of ID and OOD distribution
            can be chosen which label use as positive, defult is for ID data.
            set to False the parameter "positive_reversed" for abnormality detection (OOD).
        """
        
        # create the array with the labels initally full of zeros
        target = np.zeros((id_data.shape[0] + ood_data.shape[0]), dtype= np.int32)
        
       
        # print(target.shape)
        if positive_reversed:
            target[id_data.shape[0]:] += 1
        else:
            target[:id_data.shape[0]] += 1
       
        # print(target.shape)
        predictions = np.squeeze(np.vstack((id_data, ood_data)))
        # print(predictions.shape)
        
        aupr    = round(self.compute_aupr(target, predictions)*100, 2)
        auroc   = round(self.compute_auroc(target, predictions)*100, 2)
        print("\tAUROC(%)-> {}".format(auroc))
        print("\tAUPR (%)-> {}".format(aupr))
        
        return aupr, auroc

class CIFAR_VITEA_Abnormality_module(OOD_Classifier):

    def __init__(self, classifier: CIFAR_VITEA_benchmark, model_type: str, useGPU: bool= True, batch_size = 64, balancing_mode: str = "max", \
                 id_dataset = "cifar100", ood_dataset = "mnist", att_map_mode = "residual", image_size = 224):
        """ 
            ARGS:
            - classifier (CIFAR_VITEA_benchmark): the ViT classifier + Autoencoder (Module A) that produces the input for Module B (abnormality module)
            
            - model_type (str): choose between avaialbe model for the abnrormality module:  "encoder_v3", "encoder_v4"
            
            - batch_size (str/int): the size of the batch, set defaut to use the assigned from superclass, otherwise the int size. Default is "default".
            
            - balancing_mode (string,optinal): This has sense if use_synthethid is set to True and extended_ood is set to True.
            Choose between "max and "all", max mode give a balance number of OOD same as ID, while, all produces more OOD samples than ID. Default is "max"
            
            - ood_dataset (string, optional): name of the ood dataset use for OOD detection, the ID dataset can be CIFAR10 or CIFAR100. Use on among these:
            "cifar10","cifar100","mnist","fmnist", "svhn", "dtd", "tiny_imagenet".
            
            - att_map_mode (string, optional): select the strategy used to handle attention information. Possible values are:
            -- "residual", use residual as: (cls_attention_map - rec_cls_attention_map)^2.
            -- "cls_attention_map", use only the cls_attention_map as latent attention representation
            -- "cls_rec_attention_maps" use both cls_attention_map and its reconstruction (stacked)
            -- "full_cls_attention_maps", use cls_attention_map, and the attention map over patches (stacked)
            -- "full_cls_rec_attention_maps", use cls_attention_map, its reconstruction and the attention map over patches (stacked)
            -- "residual_full_attention_map", use residual and attention map of pathes (stacked)
            Defaults to "residual" 
            
            - image_size (int, optional): spatial dimension for the input images. Defaults is 224
        """
        super(CIFAR_VITEA_Abnormality_module, self).__init__(useGPU=useGPU)
        
        # set the classifier (module A)
        self.classifier  = classifier
        self.model_type = model_type.strip().lower()
        self.att_map_mode = att_map_mode
        
        self.name               = "Abnormality_module_ViT"
        self.image_size         = image_size
        
        # configuration variables for abnormality module (module B)
        self.augment_data_train = False
        self.ood_dataset        = ood_dataset
        self.id_dataset         = id_dataset
        self.loss_name          = "weighted bce"   # binary cross entropy or sigmoid cross entropy (weighted)
        
        # instantiation aux elements
        self.bce     = F.binary_cross_entropy_with_logits   # performs sigmoid internally
        # self.ce      = F.cross_entropy()
        self.sigmoid = F.sigmoid
        self.softmax = F.softmax
        self._build_model()
            
        # hyperparameters
        self.batch_size             = int(batch_size)
        self.lr                     = 1e-3  # 1e-3, 1e-4
        self.n_epochs               = 20    # 20, 50, 75
        self.weight_decay           = 1e-3                  # L2 regularization term 
        
        if not(balancing_mode in ["max", "all"]):
            raise ValueError('Wrong selection for balancing mode. Choose between "max" or "all".') 
        
        self.balancing_mode = balancing_mode
        
        # Define sets
        self._prepare_data_synt(verbose = True)
        
        self.path_models    = "./models/benchmarks"
        self.path_results   = "./results/benchmarks"
        
    def _build_model(self):
        # compute shapes for the input
        x_shape = (1, *self.classifier.model.input_shape)
        x = T.rand(x_shape).to(self.device)
        out = self._forward_A(x)
        
        probs_softmax       = out["probabilities"]
        encoding            = out["encoding"]
        residual_flatten    = out["residual"]


        if self.model_type == "encoder_v3":
            self.model = Abnormality_module_Encoder_ViT_v3(probs_softmax.shape, encoding.shape, residual_flatten.shape)
        elif self.model_type == "encoder_v4":
            self.model = Abnormality_module_Encoder_ViT_v4(probs_softmax.shape, encoding.shape, residual_flatten.shape)
        else:
            raise ValueError("the model type for the ViT abnormality module is not valid.")
        
        self.model.to(self.device)
        self.model.eval()
    
    def _get_ID_dataset(self, train:bool , transform_ood:bool):
        """ available OOD dataset: "cifar10","cifar100 """
        
        if self.id_dataset == "cifar10":
            data = getCIFAR10_dataset(train = train, ood_synthesis=transform_ood, resolution= self.image_size)
        elif self.id_dataset == "cifar100":
            data = getCIFAR100_dataset(train = train, ood_synthesis=transform_ood, resolution= self.image_size) 
        else:
            raise ValueError("ID dataset name is not valid")
        return data
    
    def _get_OOD_dataset(self, train:bool): #
        """ available OOD dataset: "cifar10","cifar100","mnist","fmnist", "svhn", "dtd", "tiny_imagenet"."""
        
        if self.id_dataset == "cifar10":
            data = getCIFAR10_dataset(train = train, resolution= self.image_size)
        elif self.id_dataset == "cifar100":
            data = getCIFAR100_dataset(train = train, resolution= self.image_size) 
        elif self.id_dataset == "mnist":
            data = getMNIST_dataset(train = train, resolution= self.image_size) 
        elif self.id_dataset == "fmnist":
            data = getFMNIST_dataset(train = train, resolution= self.image_size)
        elif self.id_dataset == "svhn":
            data = getSVHN_dataset(train = train, resolution= self.image_size)
        elif self.id_dataset == "dtd":
            data = getDTD_dataset(train = train, resolution= self.image_size)
        elif self.id_dataset == "tiny_imagenet":
            data = getTinyImageNet_dataset(train = train, resolution= self.image_size)
        else:
            raise ValueError("OOD dataset name is not valid")
        return data
        
    def _prepare_data_synt(self,verbose = False):
            
        """ 
            method used to prepare Dataset class used for both training and testing, synthetizing OOD data for training
            ARGS:
            - verbose (boolean, optional): choose to print extra information while loading the data
        """
        
        # synthesis of OOD data (train and valid)
        print("\n\t\t[Loading OOD (synthetized) data]\n")
        ood_data_train_syn    = self._get_ID_dataset(train=True, transform_ood=True)
        tmp        = self._get_ID_dataset(train=False, transform_ood=True)
        ood_data_valid_syn , _      = sampleValidSet(trainset = ood_data_train_syn, testset= tmp, useOnlyTest = True, verbose = True)
        
        # fetch ID data (train, valid and test)
        print("\n\t\t[Loading ID data]\n")
        id_data_train      = self._get_ID_dataset(train=True, transform_ood=False)
        tmp            = self._get_ID_dataset(train=False, transform_ood=False)
        id_data_valid , id_data_test   = sampleValidSet(trainset = id_data_train, testset= tmp, useOnlyTest = True, verbose = True)
        
        if verbose:
            print("length ID dataset  (train) -> ",  len(id_data_train))
            print("length ID dataset  (valid) -> ",  len(id_data_valid))
            print("length ID dataset  (test) -> ", len(id_data_test))
            print("length OOD dataset (train) synthetized -> ", len(ood_data_train_syn))
            print("length OOD dataset (valid) synthetized -> ", len(ood_data_valid_syn))

            
        # train set: id data train + synthetic ood (id data train transformed in ood)
        self.dataset_train = OOD_dataset(id_data_train, ood_data_train_syn, balancing_mode= self.balancing_mode)
            
        ood_data_test  = self._get_OOD_dataset(train=False)
        if verbose: print("length OOD dataset (test) -> ", len(ood_data_test))
        # test set: id data test + ood data test
        self.dataset_test  = OOD_dataset(id_data_test , ood_data_test,  balancing_mode= self.balancing_mode)

        # valid set: id data valid + synthetic ood (id data train transformed in ood)
        self.dataset_valid = OOD_dataset(id_data_valid, ood_data_valid_syn, balancing_mode= self.balancing_mode)
        
        if verbose: print("length full dataset (train/valid/test) with balancing -> ", len(self.dataset_train), len(self.dataset_valid), len(self.dataset_test))
        print("\n")
    
    def load(self, name_folder_abn, epoch):
        
        print("\n\t\t[Loading model]\n")
        
        # save folder of the train (can be used to save new files in models and results)
        self.train_name     = name_folder_abn
        self.modelEpochs    = epoch
        
        # get full path to load
        path2model          = os.path.join(self.classifier.path2classifier,  name_folder_abn, str(epoch) + ".ckpt")

        try:
            loadModel(self.model, path2model)
            self.model.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except Exception as e:
            print(e)
            print("No model: {} found for the epoch: {} in the folder: {}".format(name_folder, epoch, path2model))
    
    #           forward methods 
    def _forward_A(self, x, verbose = False, show = False):
        """ this method return a dictionary with the all the outputs from the model branches
            keys: "probabilities", "encoding", "residual"
        """
        
        # if show:showImage(x[1], "original")
            
        output_model = self.classifier.model.forward(x)  # logits, encoding, att_map
        
        # unpack the output based on the model
        logits              = output_model[0]
        encoding            = output_model[1]
        att_maps            = output_model[2]
        att_maps_patches    = output_model[3]
        
        
        probs_softmax = T.nn.functional.softmax(logits, dim=1)
        # if show: showImage(att_map[1], "attention_map", has_color=False)
        
        output = {"probabilities": probs_softmax, "encoding": encoding}
        
        if verbose: 
            print("prob shape -> ", probs_softmax.shape)
            print("encoding shape -> ",encoding.shape)
        
        if self.att_map_mode == "residual":
            # from reconstuction to squared residual
            rec_att_maps = self.classifier.autoencoder.forward(att_maps)    # generate att_map from autoencoder
            residual = T.square(rec_att_maps - att_maps)
            output["residual"]      = residual
        elif self.att_map_mode == "cls_attention_map":
            # simply use just the attention heatmap information
            output["residual"]      = att_maps        # the key name should be changed to avoid confusion on variable data
        
        elif self.att_map_mode == "cls_rec_attention_maps": 
            rec_att_maps = self.classifier.autoencoder.forward(att_maps)    # generate att_map from autoencoder
            output["residual"] = T.cat((att_maps, rec_att_maps),dim=1)
            
        elif self.att_map_mode == "full_cls_attention_maps":
            output["residual"] = T.cat((att_maps, att_maps_patches),dim=1)
            
        elif self.att_map_mode == "full_cls_rec_attention_maps":
            rec_att_maps = self.classifier.autoencoder.forward(att_maps)    # generate att_map from autoencoder
            output["residual"] = T.cat((att_maps, rec_att_maps, att_maps_patches),dim=1)
        
        elif  self.att_map_mode == "residual_full_attention_map":
            rec_att_maps = self.classifier.autoencoder.forward(att_maps)    # generate att_map from autoencoder
            residual = T.square(rec_att_maps - att_maps)
            output["residual"] = T.cat((residual, att_maps_patches),dim=1)
        else:
            raise ValueError("The attention forwarding mode selected is incompatible")
        
        # print(output["residual"].shape)

        if verbose: 
            print("residual shape ->", residual.shape)
        
        return output
                
    def _forward_B(self, probs_softmax, encoding, residual, verbose = False):
        """ if self.use_confidence is True probs_softmax should include the confidence value (Stacked) """
        
        risk_logit = self.model.forward(probs_softmax, encoding, residual)
        if verbose: print("risk logit shape", risk_logit.shape)
        return risk_logit
        
    def forward(self,x):
        
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
        
        out = self._forward_A(x)

        probs_softmax       = out["probabilities"]
        encoding            = out["encoding"]
        residual            = out["residual"]
                
        logit = self._forward_B(probs_softmax, encoding, residual)
        out   = self.sigmoid(logit)
        
        return  out
    
    #          training  and testing
    def valid(self, epoch, valid_dl):
        """
            validation method used mainly for the Early stopping training
        """
        print (f"Validation for the epoch: {epoch} ...")
        
        # set temporary evaluation mode and empty cuda cache
        self.model.eval()
        T.cuda.empty_cache()
        
        # list of losses
        losses = []

        
        for idx, (x,y) in tqdm(enumerate(valid_dl)):
            
            x = x.to(self.device)
            # y = y.to(self.device).to(T.float32)
            
            # take only label for the positive class (fake)
            y = y[:,1]
            
            # compute monodimensional weights for the full batch
            # weights = T.tensor([self.pos_weight_labels[elem] for elem in y ]).to(self.device)
            pos_weight  = T.tensor(self.pos_weight_labels).to(self.device)
            
            y = y.to(self.device).to(T.float32)

            
            with T.no_grad():
                # with autocast():
                out = self._forward_A(x)

                probs_softmax       = out["probabilities"]
                encoding            = out["encoding"]
                residual            = out["residual"]
                

                with autocast():
                    logit = T.squeeze(self.model.forward(probs_softmax, encoding, residual))
                    loss = self.bce(input=logit, target=y, pos_weight=pos_weight)   # logits bce version, peforms first sigmoid and binary cross entropy on the output
                    
                    if T.isnan(loss):
                        continue
                    
                losses.append(loss.item())

                        
        # go back to train mode 
        self.model.train()
        
        # return the average loss
        try:
            loss_valid = sum(losses)/len(losses)
        except Exception as e:
            print(e)
            loss_valid = 1
            
        print(f"Loss from validation: {loss_valid}")
        return loss_valid
    
    # self.path2classifier         
    # self.path2results_classifier
    
    @duration
    def train(self, additional_name = "", test_loop = False):
        # """ requried the ood data name to recognize the task """
        
        # compose train name
        current_date        = date.today().strftime("%d-%m-%Y")   
        train_name          = self.name + "_" + self.model_type + "_"+ additional_name + "_" + current_date
        self.train_name     = train_name
        
        # get paths to save files on model and results
        path_save_model     = os.path.join(self.classifier.path2classifier, train_name)
        path_save_results   = os.path.join(self.classifier.path2results_classifier, train_name)
        
        check_folder(path_save_model)
        check_folder(path_save_results)

        # print(path_save_model)
        # print(path_save_results)
        
        # 2) prepare the training components
        self.model.train()
        
        # compute the weights for the labels
        self.pos_weight_labels = self.compute_class_weights(verbose=True, positive="ood", only_positive_weight= True)
        
        train_dl = DataLoader(self.dataset_train, batch_size= self.batch_size,  num_workers = 8,  shuffle= True,   pin_memory= False)
        valid_dl = DataLoader(self.dataset_valid, batch_size= self.batch_size,  num_workers = 8,  shuffle = False,  pin_memory= False) 
        
        # compute number of steps for epoch
        n_steps = len(train_dl)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        # self.scheduler = None
        scaler = GradScaler()
        
        # intialize data structure to keep track of training performance
        loss_epochs     = []
        valid_history   = []
        
        # define best validation results
        best_valid          = math.inf
        best_valid_epoch    = 0
        # initialize ditctionary best models
        best_model_dict     = copy.deepcopy(self.model.state_dict())
        
        # learned epochs by the model initialization
        self.modelEpochs = 0
        
        # 3) learning loop
        
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            # print(self.classifier.model.large_encoding)
            # define cumulative loss for the current epoch and max/min loss
            loss_epoch = 0; max_loss_epoch = 0; min_loss_epoch = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dl), total= n_steps):
                
                # showImage(x[0], name=str(y[0,1]))
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # if step_idx == 20: break
                # zeroing the gradient
                self.optimizer.zero_grad()
                
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                # x.requires_grad_(True)
                y = y[:,1]                           # take only label for the positive class (fake)
                
                # compute weight for the positive class
                # pos_weight  = T.tensor([self.weights_labels[1]]).to(self.device)
                pos_weight  = T.tensor(self.pos_weight_labels).to(self.device)
                # print(pos_weight.shape)

                # int2float and move data to GPU mem                
                y = y.to(self.device).to(T.float32)               # binary int encoding for each sample
            
                # model forward and loss computation
            
                with T.no_grad():  # avoid storage gradient for the classifier
                    out = self._forward_A(x)

                    probs_softmax       = out["probabilities"]
                    encoding            = out["encoding"]
                    residual            = out["residual"]
                    
                
                # with autocast():
                                    
                probs_softmax.requires_grad_(True)
                encoding.requires_grad_(True)
                residual.requires_grad_(True)
                
                with autocast():
                    logit = T.squeeze(self.model.forward(probs_softmax, encoding, residual))
                    loss = self.bce(input=logit, target= y, pos_weight=pos_weight)
                # print(loss)

                
                loss_value = loss.item()
                if loss_value>max_loss_epoch    : max_loss_epoch = round(loss_value,4)
                if loss_value<min_loss_epoch    : min_loss_epoch = round(loss_value,4)
                
                # update total loss    
                loss_epoch += loss.item()   # from tensor with single value to int and accumulation
                
                # loss backpropagation
                scaler.scale(loss).backward()
                # loss.backward()
                                
                # compute updates using optimizer
                scaler.step(self.optimizer)
                # self.optimizer.step()

                # update weights through scaler
                scaler.update()
                
                # lr scheduler step 
                self.scheduler.step()
            
            # T.cuda.empty_cache()

            
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # include validation here if needed
            criterion = self.valid(epoch=epoch_idx+1, valid_dl = valid_dl)
            # criterion = 0
            
            valid_history.append(criterion)
            
            if criterion < best_valid:
                best_valid = criterion
                best_model_dict = copy.deepcopy(self.model.state_dict())
                best_valid_epoch = epoch_idx+1
                print("** new best abnormality module **")


            # test epochs loop for debug   
            if test_loop and last_epoch == 5: break
            
            self.modelEpochs = epoch_idx +1
         
        # 4) Save section
        
        # save loss 
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        name_valid_file         = "valid_" + str(last_epoch) + '.png' 

        plot_loss(loss_epochs, title_plot= "OOD detector", path_save = os.path.join(path_save_results, name_loss_file))
        plot_loss(loss_epochs, title_plot= "OOD detector", path_save = os.path.join(path_save_model  ,name_loss_file), show=False)  # just save, not show
        plot_valid(valid_history, title_plot= "OOD detector loss", path_save = os.path.join(path_save_results, name_valid_file))
        plot_valid(valid_history, title_plot= "OOD detector loss", path_save = os.path.join(path_save_model  ,name_valid_file), show=False)  # just save, not show


        # save model
        print("\n\t\t[Saving model]\n")
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_save_model, name_model_file)  # path folder + name file
        
        name_best_model_file    = str(best_valid_epoch) +'.ckpt'
        path_best_model_save    = os.path.join(path_save_model, name_best_model_file)
        
        
        saveModel(self.model, path_model_save)
        saveModel(best_model_dict, path_best_model_save, is_dict= True)
    
    def test_risk(self):
            
            # saving folder path
            path_results_folder         = os.path.join(self.classifier.path2results_classifier, self.train_name)
            
            # 1) prepare meta-data
            self.model.eval()
            
            # 2) prepare test
            test_dl = DataLoader(self.dataset_test, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
            
            # compute number of steps for epoch
            n_steps = len(test_dl)
            print("Number of steps per epoch: {}".format(n_steps))
            
            # empty lists to store results
            pred_risks = np.empty((0,1), dtype= np.float32)
            dl_labels = np.empty((0,2), dtype= np.int32)   
            
            for idx, (x,y) in tqdm(enumerate(test_dl), total= n_steps):
                
                # to test
                # if idx >= 1: break
                
                x = x.to(self.device)
                with T.no_grad():
                    # with autocast():
                        
                    out = self._forward_A(x)

                    probs_softmax       = out["probabilities"]
                    encoding            = out["encoding"]
                    residual            = out["residual"]
                                        
                    # if not basic Abnromality model, do recuction here
                    logit = self._forward_B(probs_softmax, encoding, residual)
                    # risk = T.squeeze(risk)
                    risk =self.sigmoid(logit)
                        
                # to numpy array
                risk   = risk.cpu().numpy()
                y       = y.numpy()
                
                pred_risks = np.append(pred_risks, risk, axis= 0)
                dl_labels = np.append(dl_labels, y, axis= 0)
                
            # divide id risk from  ood risk
            
            id_labels     =  dl_labels[:,0]                # filter by label column
            ood_labels    =  dl_labels[:,1]
            risks_id      = pred_risks[id_labels == 1]         # split forward probabilites between ID adn OOD, still a list of probabilities for each class learned by the model
            risks_ood     = pred_risks[ood_labels == 1]
            
            risks_id = risks_id[:risks_ood.shape[0]]
            # print(risks_id.shape)
            # print(risks_ood.shape)
            
            # compute statistical moments from risks output
            
            conf_all           = np.average(pred_risks)
            
            id_mean_r          =  np.mean(risks_id)
            id_std_r           =  np.std(risks_id)
            # id_entropy         = self.entropy(risks_id)
            # id_mean_e          = np.mean(id_entropy)
            # id_std_e           = np.std(id_entropy) 
        
            ood_mean_r         =  np.mean(risks_ood)
            ood_std_r          =  np.std(risks_ood)
            # ood_entropy        = self.entropy(risks_ood)
            # ood_mean_e         = np.mean(ood_entropy)
            # ood_std_e          = np.std(ood_entropy)

            # in-out of distribution moments
            print("In-Distribution risk                 [mean (confidence ID),std]  -> ", id_mean_r, id_std_r)
            print("Out-Of-Distribution risk             [mean (confidence OOD),std] -> ", ood_mean_r, ood_std_r)
            
            # print("In-Distribution Entropy risk         [mean,std]                  -> ", id_mean_e, id_std_e)
            # print("Out-Of-Distribution Entropy risk     [mean,std]                  -> ", ood_mean_e, ood_std_e)
        
            # normality detection
            print("Normality detection:")
            norm_base_rate = round(100*(risks_id.shape[0]/(risks_id.shape[0] + risks_ood.shape[0])),2)
            print("\tbase rate(%): {}".format(norm_base_rate))
            # print("\tKL divergence (entropy)")
            # kl_norm_aupr, kl_norm_auroc = self.compute_curves(id_entropy, ood_entropy)
            print("\tPrediction probability")
            p_norm_aupr, p_norm_auroc = self.compute_curves(1- risks_id, 1- risks_ood)
            
            # abnormality detection
            print("Abnormality detection:")
            abnorm_base_rate = round(100*(risks_ood.shape[0]/(risks_id.shape[0] + risks_ood.shape[0])),2)
            print("\tbase rate(%): {}".format(abnorm_base_rate))
            # print("\tKL divergence (entropy)")
            # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(-id_entropy, -ood_entropy, positive_reversed= True)
            # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-id_entropy, 1-ood_entropy, positive_reversed= True)
            print("\tPrediction probability")
            p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(risks_id, risks_ood, positive_reversed= True)
            # p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-risks_id, 1-risk_ood, positive_reversed= True)
            

            
            # compute fpr95, detection_error and threshold_error 
            metrics_norm = self.compute_metrics_ood(1-risks_id, 1-risks_ood)
            print("OOD metrics:\n", metrics_norm)  
            metrics_abnorm = self.compute_metrics_ood(risks_id, risks_ood, positive_reversed= True, path_save = path_results_folder, epoch= self.modelEpochs)
            print("OOD metrics:\n", metrics_abnorm)  
            
            
                    # store statistics/metrics in a dictionary
            data = {
                "ID_max_prob": {
                    "mean":  float(id_mean_r), 
                    "var":   float(id_std_r)
                },
                "OOD_max_prob": {
                    "mean": float(ood_mean_r), 
                    "var":  float(ood_std_r) 
                },
                # "ID_entropy":   {
                #     "mean": float(id_mean_e), 
                #     "var":  float(id_std_e)
                # },
                # "OOD_entropy":  {
                #     "mean": float(ood_mean_e), 
                #     "var":  float(ood_std_e)
                # },
                
                "normality": {
                    "base_rate":        float(norm_base_rate),
                    # "KL_AUPR":          float(kl_norm_aupr),
                    # "KL_AUROC":         float(kl_norm_auroc),
                    "Prob_AUPR":        float(p_norm_aupr),   
                    "Prob_AUROC":       float(p_norm_auroc)
                },
                "abnormality":{
                    "base_rate":        float(abnorm_base_rate),
                    # "KL_AUPR":          float(kl_abnorm_aupr),
                    # "KL_AUROC":         float(kl_abnorm_auroc),
                    "Prob_AUPR":        float(p_abnorm_aupr),   
                    "Prob_AUROC":       float(p_abnorm_auroc),

                },
                "avg_confidence":               float(conf_all),
                "fpr95_normality":              float(metrics_norm['fpr95']),
                "detection_error_normality":    float(metrics_norm['detection_error']),
                "threshold_normality":          float(metrics_norm['thr_de']),
                "fpr95_abnormality":            float(metrics_abnorm['fpr95']),
                "detection_error_abnormality":  float(metrics_abnorm['detection_error']),
                "threshold_abnormality":        float(metrics_abnorm['thr_de'])
            }
            
            # save data (JSON)
            name_result_file            = 'metrics_ood_{}.json'.format(self.modelEpochs)
            path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
            
            print(path_result_save)
            saveJson(path_file = path_result_save, data = data)
    

if __name__ == "__main__":
    #                           [Start test section] 
    
    # 1) pytorch implementation
    def train_baseline_T():            
        classifier = MNISTClassifier()
        classifier.train()
        classifier.load()
    
    # tensorflow implementation
    def train_baseline_tf():
        classifier = MNISTClassifier_keras()
        classifier.train()
        classifier.load_model()
    
    # train decoders
    def train_ResNetED_content():
        model = Decoder_ResNetEDS(scenario="content")
        model.train(name_train="faces_resnet50ED")
    
    def train_Unet_content(): 
        model = Decoder_Unet(scenario="content")
        model.train(name_train="faces_Unet4")
    
    # show results from end-dec
    def showReconstructionResnetED(name_model, epoch, scenario):
        
        n_picture = 0
        model = Decoder_ResNetEDS(scenario = scenario, useGPU= True)
        model.load(name_model, epoch)
        img, _ = model.train_dataset.__getitem__(n_picture)
        showImage(img, name = "original_"+ str(n_picture), save_image = True)
        img     = T.unsqueeze(img, dim= 0).to(model.device)
        enc     = model.model.encoder_module.forward(img)
        rec_img = model.model.decoder_module.forward(enc)
        rec_img = T.squeeze(rec_img, dim = 0)
        showImage(rec_img, name="reconstructed_"+ str(n_picture) + "_decoding_" + name_model, save_image = True)
        
    def showReconstructionUnet(name_model, epoch, scenario):
        n_picture = 3
        decoder = Decoder_Unet(scenario = scenario, useGPU= True)
        decoder.load(name_model, epoch)
        img, _ = decoder.train_dataset.__getitem__(n_picture)
        showImage(img, name = "original_"+ str(n_picture), save_image = True)
        img     = T.unsqueeze(img, dim= 0).to(decoder.device)
        
        rec_img, _ = decoder.model.forward(img)
        rec_img = T.squeeze(rec_img, dim = 0)
        
        showImage(rec_img, name="reconstructed_"+ str(n_picture) + "_decoding_" + name_model, save_image = True)
    
    # test ViTEA
    def train_ViT_Cifar():
        classifier = CIFAR_ViT_Classifier()
        # classifier.train()
        classifier.load()
        classifier.test_accuracy()
    
    def train_VITEA_CIFAR():
        classifier = CIFAR_ViTEA_Classifier(prog_model = 3, cifar100=False)
        classifier.train()
        # classifier.load()
        classifier.test_accuracy()
        
    def test_attention_map():
        classifier =  CIFAR_ViTEA_Classifier(prog_model = 3, cifar100=False)
        classifier.load()
        data_iter = classifier.train_data
        save    = False
        img_id  = 0
        
        img, y = data_iter.__getitem__(img_id)

            
        showImage(img, save_image= save, name="attention_original_" + str(img_id))

        
        img = img.unsqueeze(dim=0)
        print(img.shape)
        
        
        img = img.to(device = classifier.device)
        
        _, _, att_map = classifier.model.forward(img)
        
        print(att_map.shape, T.max(att_map), T.min(att_map))
        
    
        showImage(att_map[0], has_color= False, save_image= save, name="attention_map_" + str(img_id))

        # show_imgs_blend(img_d, att_map.cpu(), alpha=0.8, save_image= save, name="attention_blend_" + str(img_id))
        result, att_map  = include_attention(img, att_map, alpha= 0.7)
        
        print(att_map.shape, T.max(att_map), T.min(att_map))
        
        print(result.shape)
        
        showImage(result[0], save_image= save, name="attention_map_" + str(img_id))
    
    #                           [Benchmark functions]
    
    #                               train & test classifier
    def train_classifier_benchmark(add2name, prog_model = 3, cifar100 = False, image_size = 224):
        
        vitea = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model=prog_model, image_size = image_size)
        vitea.train_classifier(add_in_name= add2name)
        # vitea.test(epoch=50, name_folder= "train_50_epochs_22-02-2024")
    
    def continue_train_classifier_benchmark(name_folder, epoch_start, end_epoch, prog_model = 3, cifar100 = False, image_size = 224):
        vitea = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model=prog_model, image_size=image_size)
        vitea.load_classifier(name_folder=name_folder, epoch=epoch_start)
        vitea.train_classifier(start_epoch=epoch_start, end_epoch= end_epoch)
        # vitea
        
    def train_AE_benchmark(name_folder, epoch_classifier, prog_model = 3, cifar100 = False, image_size = 224):
        vitea = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model= prog_model, image_size= image_size)
        vitea.load_classifier(name_folder=name_folder, epoch= epoch_classifier)
        vitea.train_ae(name_folder=name_folder)
        
    def test_classifier_benchmark(name_folder, epoch_classifier, prog_model = 3, cifar100 = False, image_size = 224):
        vitea = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model= prog_model, image_size=image_size)
        vitea.test(name_folder= name_folder, epoch=epoch_classifier)   # load is peformed directly in the test function
    
    #                               train & test abn module
    
    # load classifier here
    choose_model = None
    
    if choose_model == 0:
        cifar100 = False
        prog_model = 3
        name_folder = "train_DeiT_tiny_22-02-2024"
        epoch_classifier = 100
        epoch_autoencodder = None
        classifier = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model= prog_model)
        classifier.load_classifier(epoch=epoch_classifier, name_folder=name_folder)
        classifier.load_autoencoder(epoch=epoch_autoencodder, name_folder= name_folder)
    
    def train_abn_module(model_type = "encoder_v3", image_size = 224):
        abn = CIFAR_VITEA_Abnormality_module(classifier, model_type= model_type, image_size= image_size)
        abn.train(additional_name= "test")
        
    def test_abn_module(name_folder, epoch, model_type= "encoder_v3", image_size = 224): 
        abn = CIFAR_VITEA_Abnormality_module(classifier, model_type= model_type, image_size= image_size)
        abn.load(name_folder_abn = name_folder, epoch= epoch)
        abn.test_risk()
    

    
    """                              train cifar 10                                         """
    
    # train_classifier_benchmark("DeiT_tiny", prog_model = 3)
    # test_classifier_benchmark("train_DeiT_tiny_22-02-2024", 100, prog_model=3, cifar100=False)
    
    # train_classifier_benchmark("DeiT_small", prog_model = 2)
    # continue_train_classifier_benchmark("train_DeiT_small_23-02-2024", epoch_start=100, end_epoch=150, prog_model=2)
    # test_classifier_benchmark("train_DeiT_small_23-02-2024", 150, prog_model=2, cifar100=False)
    
    # train_classifier_benchmark("2_DeiT_tiny", prog_model = 3)
    # continue_train_classifier_benchmark("train_2_DeiT_tiny_26-02-2024", epoch_start=50, end_epoch=150, prog_model=3)
    
    
    train_classifier_benchmark("3_DeiT_tiny", prog_model = 3)
    
    """                            train abn module cifar 10                                """
    
    # train_abn_module()
    # test_abn_module("Abnormality_module_ViT_encoder_v3_test_24-02-2024", 4)
    
    
    #                           [End test section] 
    
    """ 
            Past test/train launched: 
        
    train_baseline_tf()
   
    train_ResNetED_content()
    train_Unet_content()
    train_ResNetED_content()
    
    showReconstructionResnetED(name_model="faces_resnet50ED_18-11-2023", epoch= 40, scenario = "content")
    showReconstruction(name_model="faces_resnet50ED_18-11-2023", epoch= 40, scenario = "content")
    """