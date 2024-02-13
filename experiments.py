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
from    torch.optim                         import Adam, lr_scheduler
from    torch.cuda.amp                      import GradScaler, autocast
from    torch.utils.data                    import DataLoader


# local modules
from    dataset                             import getMNIST_dataset, getCIFAR100_dataset, getCIFAR10_dataset,CDDB_binary_Partial
from    models                              import FC_classifier, get_fc_classifier_Keras, ResNet_EDS, Unet4, ViT_b16_ImageNet, \
                                                    ViT_timm_EA
from    utilities                           import duration, saveModel, loadModel, showImage, check_folder, plot_loss, image2int, \
                                                    ExpLogger, trans_input_v2, show_imgs_blend, include_attention


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
                                                ViT attention Experiment
#####################################################################################################################
"""




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
        
    
    test_attention_map()
    
    
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