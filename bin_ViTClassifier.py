from    time                                import time
import  random              
from    tqdm                                import tqdm
from    datetime                            import date, datetime
import  math
import  torch                               as T
import  numpy                               as np
import  os              
import  copy
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from    torch.nn                            import functional as F
from    torch.optim                         import Adam, lr_scheduler
from    torch.cuda.amp                      import GradScaler, autocast
from    torch.utils.data                    import DataLoader
from    torch.autograd                      import Variable
from    torchvision.transforms              import v2
from    torch.utils.data                    import default_collate
from    PIL                                 import Image
# Local imports

from    utilities                           import plot_loss, plot_valid, saveModel, metrics_binClass, loadModel, sampleValidSet, \
                                            duration, check_folder, cutmix_image, showImage, image2int, ExpLogger, alpha_blend_pytorch,\
                                            show_imgs_blend, include_attention, saveJson
from    dataset                             import getScenarioSetting, CDDB_binary, CDDB_binary_Partial
from    models                              import ViT_b_scratch, ViT_b16_ImageNet, ViT_timm, ViT_timm_EA,\
                                                AutoEncoder, AutoEncoder_v2, VAE
from    bin_classifier                      import BinaryClassifier



class DFD_BinViTClassifier_v6(BinaryClassifier):
    """
        binary classifier for deepfake detection using partial CDDB dataset for the chosen scenario configuration.
        Model used: Vision-Transformer
        This class is a modification of DFD_BinClassifier_v5 class in bin_Classifier module to work with ViTs classifier,
        no confidence and no reconstruction is included here
        
        

    """
    def __init__(self, scenario, useGPU = True, patch_size = None, emb_size = None,  batch_size = 32,
                 model_type = "ViT_base", transform_prog = 0):  # batch_size = 32 or 64
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
            patch_size (int, optional): size used to divide the image in patches. Defaults to None (to use default settings)
            emb_size (int, optional): size used to build the token embeddings and the internal encodings of the transformer. Defaults to None (to use default settings)
            batch_size (int, optional): batch size used by dataloaders, Usually 32 or 64 (for lighter models). Defaults is 32.
            model_type (str, optional): choose the Unet architecture between :
                - "ViT_base_[size:xs,s,m][patch_size]"
                - "ViT_b16_pretrained"   (imagenet)
                - "ViT_pretrained_timm"
            Defaults is "ViT_base". 
            transform_prog (int, optional). Select the prog for input data trasnformation.Defaults is 1.
        """
        super(DFD_BinViTClassifier_v6, self).__init__(useGPU = useGPU, batch_size = batch_size, model_type = model_type)
        self.version = 6
        self.scenario = scenario
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.augment_data_train = True
        self.use_cutmix         = False     # problem when we are learning transformer over sequence of patches
        self.n_classes          = 2
        self.transform_prog   = transform_prog
        
        model_type_check = model_type.lower().strip()
        
        # prepare args
        kwargs = {"n_classes": 2}
        if self.patch_size is not(None):
            kwargs["patch_size"] = self.patch_size
        if self.emb_size is not(None):
            kwargs["emb_size"] = self.emb_size
        
        print(kwargs)
        
        # load model
        if "vit_base" in model_type_check:
            if "_xs" in model_type_check: 
                self.model      = ViT_b_scratch(**kwargs,  n_layers= 6, n_heads=4)
            elif "_s" in  model_type_check:
                self.model      = ViT_b_scratch(**kwargs)
            elif "_m" in model_type_check:
                self.model      = ViT_b_scratch(**kwargs, n_layers = 12, n_heads=12)
        
                
                # self.model = ViT_base_2(n_classes= self.n_classes, n_layers= 10, n_heads=6)
            # elif "_m" in model_type.lower().strip():
            #      self.model = ViT_base_2(n_classes= self.n_classes, n_layers= 10, n_heads=10)

            else:
                raise ValueError("specify the dimension of the ViT_base model")
            
        elif "vit_b16_pretrained_imagenet" in model_type.lower().strip():
            self.model = ViT_b16_ImageNet(**kwargs)
            
        elif "vit_pretrained_timm" in model_type.lower().strip():
            self.model = ViT_timm(**kwargs)
        
        else:
            raise ValueError("The model type is not a Unet model")
        
        self.model.to(self.device)
        self.model.eval()
        
        self._load_data()
    
        # activation function for logits
        self.sigmoid    = T.nn.Sigmoid().cuda()
        # bce defined in the training since is necessary to compute the labels weights 

        # learning hyperparameters (default)
        self.learning_coeff         = 0.3                                             # multiplier that increases the training time
        self.lr                     = 1e-4    # 1e-3 or 1e-4
        self.n_epochs               = math.floor(50 * self.learning_coeff)
        self.start_early_stopping   = math.floor(self.n_epochs/2)                       # epoch to start early stopping
        self.weight_decay           = 1e-3                                              # L2 regularization term 
        self.patience               = max(math.floor(5 * self.learning_coeff),5)        # early stopping patience
        self.early_stopping_trigger = "loss"                                            # values "acc" or "loss"
        
        # loss definition + interpolation values for the new loss
        self.loss_name              = "weighted bce"
        self._check_parameters()
        
        # training components definintion
        self.optimizer_name     = "Adam"
        self.lr_scheduler_name  = "ReduceLROnPlateau"
        self.optimizer = Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
    def _load_data(self):
        # load dataset: train, validation and test.
        
        # get train set
        print(f"\n\t\t[Loading CDDB binary partial ({self.scenario}) data]\n")

        self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False,
                                                  augment= self.augment_data_train, label_vector= True, type_transformation= self.transform_prog)
        
        # get valid and test sets
        test_dataset        = CDDB_binary_Partial(scenario = self.scenario, train = False, ood = False,
                                                  augment= False, type_transformation= self.transform_prog)
        self.valid_dataset, self.test_dataset = sampleValidSet(trainset= self.train_dataset, testset= test_dataset, useOnlyTest = True, verbose = True)
        
    def _check_parameters(self):
        if not(self.early_stopping_trigger in ["loss", "acc"]):
            raise ValueError('The early stopping trigger value must be chosen between "loss" and "acc"')
    
    def _hyperParams(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs_max": self.n_epochs,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.patience,
            "early_stopping_trigger": self.early_stopping_trigger,
            "early_stopping_start_epoch": self.start_early_stopping,
                }
    
    def _dataConf(self):
        
        try:    input_shape = str(self.model.input_shape)
        except: input_shape = "empty"
        
        try:    patch_size = self.model.patch_size
        except: patch_size = "empty"
        
        try:    n_channels = self.model.n_channels
        except: n_channels = "empty"
        
        try:    emb_dim = str(self.model.emb_size)
        except: emb_dim = "empty"
            
        try:    n_layers = str(self.model.n_layers)
        except: n_layers = "empty"
            
        try:    n_heads = str(self.model.n_heads)
        except: n_heads = "empty"
        
        try:    dropout_percentage= str(self.model.dropout)
        except: dropout_percentage = "empty"
        
        try:    weights_classes = self.weights_labels 
        except: weights_classes = "empty"
        
        specified_data = {
            "date_training": date.today().strftime("%d-%m-%Y"),
            "model_type": self.model_type,
            "model_class": self.model.__class__.__name__,
            "input_shape": input_shape,
            "prog_data_transformation": self.transform_prog, 
            "data_scenario": self.scenario,
            "version_train": self.version,
            "optimizer": self.optimizer_name,
            "scheduler": self.lr_scheduler_name,
            "loss": self.loss_name,
            "base_augmentation": self.augment_data_train,
            "cutmix": self.use_cutmix,            
            "grad_scaler": True,                # always true
            "labels weights": weights_classes,
            
            # model specific properties:
            "patch_dimension": str((n_channels, patch_size, patch_size)), 
            "embedding_dimension": emb_dim,
            "number_layers": n_layers,
            "number_heads": n_heads,
            "dropout_%": dropout_percentage
            }
        
        # include auto-inferred data from the model if available
        try:
            concat_dict = lambda x,y: {**x, **y}
            model_data = self.model.getAttributes()
            return concat_dict(specified_data, model_data)
        except:
            return specified_data
        
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
                logits = self.model.forward(x) 
                # pred = self.sigmoid(logits)
                
                if self.early_stopping_trigger == "loss":
                    # loss = self.bce(input=pred, target=y)   # pred bce version
                    loss = self.bce(input=logits, target=y)   # logits bce version
                    losses.append(loss.cpu().item())
                    
                elif self.early_stopping_trigger == "acc":
                    # prepare predictions and targets
                    pred = self.sigmoid(logits)
                    y_pred  = T.argmax(pred, -1).cpu().numpy()  # both are list of int (indices)
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
        
        # set the full current model name 
        self.classifier_name = name_train
        
        # define the model dir path and create the directory
        current_date = date.today().strftime("%d-%m-%Y")    
        path_model_folder       = os.path.join(self.path_models,  name_train + "_v{}_".format(str(self.version)) + current_date)
        check_folder(path_model_folder, is_model= True)
        
        # define train dataloader
        train_dataloader = None
    
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # define valid dataloader
        valid_dataloader = DataLoader(self.valid_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # model in training mode
        self.model.train()
        
        # define the optimization algorithm
        self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        # define the loss function and class weights
        
        # compute labels weights and cross entropy loss
        self.weights_labels = self.compute_class_weights(only_positive_weight=True)
        
        # self.bce       = T.nn.BCEWithLogitsLoss(weight=T.tensor(self.weights_labels)).to(device=self.device)
        self.bce       = T.nn.BCEWithLogitsLoss(pos_weight=T.tensor(self.weights_labels)).to(device=self.device)
        
        # self.bce      = T.nn.BCELoss(weight = T.tensor(self.weights_labels)).cuda()   # to apply after sigmodid 
        
        # learning rate scheduler
        if self.early_stopping_trigger == "loss":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr*0.01, verbose = True) # reduce of a half the learning rate 
        elif self.early_stopping_trigger == "acc":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr*0.01, verbose = True) # reduce of a half the learning rate 
        
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
        
        tmp_losses = []
        
        # loop over epochs
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # if epoch_idx != 2: continue

            # define cumulative loss for the current epoch and max/min loss
            loss_epoch = 0; max_loss_epoch = 0; min_loss_epoch = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
            
            printed_nan = False
                    
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= n_steps):
                
                # if epoch_idx <= 1: continue
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(True)
                y = y.to(self.device)               # binary int encoding for each sample
                y = y.to(T.float)
                
                # zeroing the gradient
                self.optimizer.zero_grad()
                
                # model forward and loss computation
                with autocast():   
                
                    logits = self.model.forward(x) 
                    
                    # apply activation function to logits
                    # pred = self.sigmoid(logits)
                        
                    # compute classification loss
                    try:
                        loss      = self.bce(input=logits, target=y)   # bce from logits (no weights loaded)
                        # loss      = self.bce(input=pred, target=y)   # classic bce with "probabilities"
                    except Exception as e :
                        print(e)
                        print("skipping, problem in the computation of the loss")
                        # loss      = T.tensor(0)
                        continue
                    
                    if T.isnan(loss).any().item() and not(printed_nan):
                        print(loss)
                        print(logits)
                        print(T.norm(logits))
                        print(y)
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                print(f'Parameter: {name}, Gradient Magnitude: {param.grad.norm().item()}')
                        
                        printed_nan = True
                        
                    try:
                        loss_value = loss.cpu().item()
                    except:
                        print(loss.shape, loss)
                        print(logits.shape)
                        print(y.shape)
                        raise ValueError("error converting tensor to scaler with .item() function")
                    
                    
                    tmp_losses.append(loss_value)
                    if step_idx%100 == 0:
                        print("norm logits every 100 epochs ->", T.norm(logits).cpu().item())
                        print("avg loss every 100 epochs ->", sum(tmp_losses)/len(tmp_losses))
                        tmp_losses = []
                    
                    if loss_value>max_loss_epoch    : max_loss_epoch = round(loss_value,4)
                    if loss_value<min_loss_epoch    : min_loss_epoch = round(loss_value,4)
                    
                    # update total loss    
                    loss_epoch += loss_value   # from tensor with single value to int and accumulation
                    
                    # loss backpropagation
                    scaler.scale(loss).backward()
                    
                    # compute updates using optimizer
                    scaler.step(self.optimizer)

                    # update weights through scaler
                    scaler.update()
                    
                    # clip gradients
                    # T.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # include validation here if needed
            criterion = self.valid(epoch=epoch_idx+1, valid_dataloader= valid_dataloader)
            valid_history.append(criterion)  
            
            # initialize not early stopping
            early_exit = False 
            
            # early stopping update
            if epoch_idx > 0 and last_epoch >= self.start_early_stopping:
                print("Early stopping step ...")
                
                if self.early_stopping_trigger == "loss":
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
                            
                elif self.early_stopping_trigger == "acc":
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

        # log GPU memory statistics during training
        logger.log_mem(T.cuda.memory_summary(device=self.device))
        
        # create path for the model save
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        # create path for the model results
        path_results_folder     = os.path.join(self.path_results, name_train + "_v{}_".format(str(self.version)) + current_date)
        check_folder(path_results_folder, is_model = True) # create if doesn't exist
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        name_valid_file         = "valid_{}_{}.png".format(self.early_stopping_trigger, str(last_epoch))
        path_lossPlot_save      = os.path.join(path_results_folder, name_loss_file)
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs        = last_epoch
        
        # save loss plot
        if test_loop:
            plot_loss(loss_epochs, title_plot= "classifier", path_save = None)
            plot_valid(valid_history, title_plot= "classifier  "+ self.early_stopping_trigger, path_save = None)
        else: 
            plot_loss(loss_epochs, title_plot= "classifier", path_save = path_lossPlot_save)
            plot_loss(loss_epochs, title_plot= "classifier", path_save = os.path.join(path_model_folder,name_loss_file), show=False)
            plot_valid(valid_history, title_plot= "classifier  "+ self.early_stopping_trigger, path_save = os.path.join(path_results_folder, name_valid_file))
            plot_valid(valid_history, title_plot= "classifier  "+ self.early_stopping_trigger, path_save = os.path.join(path_model_folder,name_valid_file), show=False)
        
        # save model
        saveModel(self.model, path_model_save)
    
        # terminate the log session
        logger.end_log(model_results_folder_path=path_results_folder)

    def test(self):
        # call same test function but don't load again the data
        super().test(load_data=False)
    
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
        
        logits = self.model.forward(x) 
        
        probs       = self.sigmoid(logits)    # change to softmax in multi-class context
        pred        = T.argmax(probs, -1)
        fake_prob   = probs[:,1]   # positive class probability (fake probability)
        
        return pred, fake_prob, logits

class DFD_BinViTClassifier_v7(BinaryClassifier):
    """
        binary classifier for deepfake detection using partial CDDB dataset for the chosen scenario configuration.
        Model used: Vision-Transformer
        This class is a modification of DFD_BinClassifier_v6 class of this module. This class use a ViT model that returns
        not only classification logits, but also an encoding of the image and the attention map.
        Together with the ViT is also trained an autoencoder to reconstruct the attention map, this is used for OOD detection inference.
    """
    def __init__(self, scenario, useGPU = True, patch_size = None, emb_size = None,  batch_size = 32, model_type = "ViTEA_timm", 
                 prog_pretrained_model= 3, model_ae_type = "VAE" , train_together = True, transform_prog = 0):  # batch_size = 32 or 64
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
            patch_size (int, optional): size used to divide the image in patches. Defaults to None (to use default settings)
            emb_size (int, optional): size used to build the token embeddings and the internal encodings of the transformer. Defaults to None (to use default settings)
            batch_size (int, optional): batch size used by dataloaders, Usually 32 or 64 (for lighter models). Defaults is 32.
            model_type (str, optional): choose the Unet architecture between :
                - "ViTEA_timm"
            Defaults is "ViTEA_timm". 
            
            model_ae_type (str, optional): choose the autoencoder architecture between :
                -"AE_v1"
                -"AE_V2"
                -"VAE"
            Defaults is "VAE". 
            
            transform_prog (int, optional). Select the prog for input data trasnformation.Defaults is 0 (normalization values btw 0 and 1). (pre-trained models can use its own trasnformation compose)
            train_together (boolean, optional). Choose whether train classifier and autoencoder together. Defaults is True
            prog_pretrained_model (int, optional). Select the prog for ViT pretrained model.Defaults is 3 (tiny DeiT).

        """
        super(DFD_BinViTClassifier_v7, self).__init__(useGPU = useGPU, batch_size = batch_size, model_type = model_type)
        self.version                = 7
        self.scenario               = scenario
        self.patch_size             = patch_size
        self.emb_size               = emb_size
        self.augment_data_train     = True
        self.use_cutmix             = False     # problem when we are learning transformer over sequence of patches
        self.n_classes              = 2
        self.transform_prog         = transform_prog
        self.external_autoencoder   = True
        self.train_together         = train_together
        self.model_prog             = prog_pretrained_model
        self.model_ae_type          = model_ae_type.lower().strip()

        # prepare args
        kwargs = {"n_classes": 2}
        if self.patch_size is not(None):
            kwargs["patch_size"] = self.patch_size
        if self.emb_size is not(None):
            kwargs["emb_size"] = self.emb_size
        

        if "vitea_timm" in model_type.lower().strip():
            self.model = ViT_timm_EA(n_classes=2, prog_model= self.model_prog)
        
        else:
            raise ValueError("The model type for the ViT classifier is unknown")
        
        #                       models
        self.model.to(self.device)
        self.model.eval()
        
        # AutoEncoder, AutoEncoder_v2, VAE
        if "ae_v1" in  self.model_ae_type:
            self.autoencoder = AutoEncoder() 
        elif "ae_v2" in self.model_ae_type: 
            self.autoencoder = AutoEncoder_v2() 
        elif "vae" in self.model_ae_type: 
            self.autoencoder = VAE() 
        else:
            raise ValueError("The model type for the autoencoder is unknown")
            
         
        self.autoencoder.to(self.device)
        self.autoencoder.eval()
        #                       data
        self._load_data()


        #                       activation and loss function
        self.sigmoid    = T.nn.Sigmoid().cuda()
        self.linear_af  = T.nn.Identity().cuda()
        
        # bce defined in the training since is necessary to compute the labels weights
        self.mse        = T.nn.MSELoss()
        self.mae        = T.nn.L1Loss()

        # learning hyperparameters common
        self.lr                     = 1e-3    # 1e-3 or 1e-4
        self.weight_decay           = 1e-3    # L2 regularization term 
        
        # learning hyperparameters ViT
        self.learning_coeff         = 1.5                                             #  [0.5, 1, 2] multiplier that increases the training time
        self.n_epochs               = math.floor(50 * self.learning_coeff)
        self.start_early_stopping   = math.floor(self.n_epochs/2)                       # epoch to start early stopping                         
        self.patience               = max(math.floor(5 * self.learning_coeff),5)        # early stopping patience
        self.early_stopping_trigger = "loss"                                            # validation metric: values "acc" or "loss"
        
        # learning hyperparameters autoencoder
        if self.train_together:
            self.n_epochs_AE             = self.n_epochs
        else:
            self.n_epochs_AE             = max(math.floor(50 * self.learning_coeff), 50)
        
        
        # loss definition + interpolation values for the new loss
        self.loss_name              = "weighted bce"
        self.loss_name_ae           = "MAE"   # or MAE
        self._check_parameters()
        
        # training components definintion
        self.optimizer_name         = "Adam"
        self.lr_scheduler_name      = "ReduceLROnPlateau"
        self.optimizer              = Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        self.optimizer_ae           = Adam(self.autoencoder.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        self.lr_scheduler_ae_name   = "lr_scheduler.OneCycleLR"
        
        self.path2model_results = os.path.join(self.path_results, getScenarioSetting()[self.scenario] + "_no_train_" + self.model.models_avaiable[self.model_prog])
        
    def _load_data(self):
        # load dataset: train, validation and test.
        
        # get train set
        print(f"\n\t\t[Loading CDDB binary partial ({self.scenario}) data]\n")

        self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False,
                                                  augment= self.augment_data_train, label_vector= True, type_transformation= self.transform_prog)
        
        # get valid and test sets
        test_dataset        = CDDB_binary_Partial(scenario = self.scenario, train = False, ood = False,
                                                  augment= False, type_transformation= self.transform_prog)
        self.valid_dataset, self.test_dataset = sampleValidSet(trainset= self.train_dataset, testset= test_dataset, useOnlyTest = True, verbose = True)
        
    def _check_parameters(self):
        if not(self.early_stopping_trigger in ["loss", "acc"]):
            raise ValueError('The early stopping trigger value must be chosen between "loss" and "acc"')
    
    def _hyperParams(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs_max (classifier)": self.n_epochs,
            "epochs_max_AE" : self.n_epochs_AE,
            "weight_decay": self.weight_decay,
            "early_stopping_patience (classifier)": self.patience,
            "early_stopping_trigger (classifier)": self.early_stopping_trigger,
            "early_stopping_start_epoch (classifier)": self.start_early_stopping,
                }
    
    def _dataConf(self):
        
        try:    input_shape = str(self.model.input_shape)
        except: input_shape = "empty"
        
        try:    patch_size = self.model.patch_size
        except: patch_size = "empty"
        
        try:    n_channels = self.model.n_channels
        except: n_channels = "empty"
        
        try:    emb_dim = str(self.model.emb_size)
        except: emb_dim = "empty"
            
        try:    n_layers = str(self.model.n_layers)
        except: n_layers = "empty"
            
        try:    n_heads = str(self.model.n_heads)
        except: n_heads = "empty"
        
        try:    dropout_percentage= str(self.model.dropout)
        except: dropout_percentage = "empty"
        
        try:    weights_classes = self.pos_weight_label 
        except: weights_classes = "empty"
        
        
        try:    path_model = self.path_model_folder
        except: path_model = "empty"
        
        try:    path_results = self.path_results_folder
        except: path_results = "empty"
        
        
        specified_data = {
            "date_training": date.today().strftime("%d-%m-%Y"),
            "model_type": self.model_type,
            "model_class": self.model.__class__.__name__,
            "autoencoder_class": self.autoencoder.__class__.__name__,
            "input_shape": input_shape,
            "ViT pretrained model": self.model.models_avaiable[self.model_prog],
            "external_autoencoder": self.external_autoencoder,
            "prog_data_transformation": self.transform_prog, 
            "data_scenario": self.scenario,
            "version_train": self.version,
            "optimizer": self.optimizer_name,
            "scheduler": self.lr_scheduler_name,
            "scheduler_autoencdoer" :self.lr_scheduler_ae_name, 
            "loss": self.loss_name,
            "loss autoencoder": self.loss_name_ae,
            "base_augmentation": self.augment_data_train,
            "cutmix": self.use_cutmix,            
            "grad_scaler": True,                # always true
            "labels weights": weights_classes,
            
            # model specific properties:
            "patch_dimension": str((n_channels, patch_size, patch_size)), 
            "embedding_dimension": emb_dim,
            "number_layers": n_layers,
            "number_heads": n_heads,
            "dropout_%": dropout_percentage,
            
            # paths
            "path2model": path_model,
            "path2results":  path_results
            }
        
        # include auto-inferred data from the model if available
        try:
            concat_dict = lambda x,y: {**x, **y}
            model_data = self.model.getAttributes()
            return concat_dict(specified_data, model_data)
        except:
            return specified_data
    
    def init_logger_ae(self, path_model, add2name = ""):
        """
            path_model -> specific path of the current model training
        """
        
        logger = ExpLogger(path_model=path_model, add2name=add2name)
        logger.write_config(self._dataConf())
        logger.write_hyper(self._hyperParams())
        try:
            logger.write_model(self.autoencoder.getSummary(verbose=False))
        except:
            print("Impossible to retrieve the model structure for logging")
        
        return logger
    
    # ------------------------------- validation 
    
    def valid(self, epoch, valid_dataloader):
        """
            validation method used mainly for the Early stopping training
        """
        print (f"Validation for the epoch: {epoch} ...")
        
        # set temporary evaluation mode and empty cuda cache
        self.model.eval()
        self.autoencoder.eval()
        T.cuda.empty_cache()
        
        # list of losses
        losses      = []
        losses_ae   = []
        
        # counters to compute accuracy
        correct_predictions = 0
        num_predictions = 0
        
        for (x,y) in tqdm(valid_dataloader):
            
            x = x.to(self.device)
            # y = y.to(self.device).to(T.float32)
            y = y.to(self.device).to(T.float32)
            
            with T.no_grad():
                
                out = self.model.forward(x) 
                logits      = out[0]
                att_maps    = out[2]
                
                
                # logits, _, att_maps  = self.model.forward(x) 
                
                # pred = self.sigmoid(logits)
                
                # classifier criterion
                if self.early_stopping_trigger == "loss":
                    # loss = self.bce(input=pred, target=y)   # pred bce version
                    loss = self.bce(input=logits, target=y)   # logits bce version
                    losses.append(loss.cpu().item())
                    
                elif self.early_stopping_trigger == "acc":
                    # prepare predictions and targets
                    pred = self.sigmoid(logits)
                    y_pred  = T.argmax(pred, -1).cpu().numpy()  # both are list of int (indices)
                    y       = T.argmax(y, -1).cpu().numpy()
                    
                    # update counters
                    correct_predictions += (y_pred == y).sum()
                    num_predictions += y_pred.shape[0]
                    
                # Autoencoder criterion
                x_ae = att_maps.clone().to(device =self.device)
                
                rec_att_map = self.autoencoder(x_ae)
                loss_ae     = self.mae(rec_att_map, x_ae)
                losses_ae.append(loss_ae.cpu().item())
                
                        
        # go back to train mode 
        self.model.train()
        self.autoencoder.train()
        
        
        loss_ae_valid = sum(losses_ae)/len(losses_ae)
        print(f"Loss from validation (autoencoder): {loss_ae_valid}")
        
        if self.early_stopping_trigger == "loss":
            # return the average loss
            loss_valid = sum(losses)/len(losses)
            print(f"Loss from validation (classifier): {loss_valid}")
            return loss_valid, loss_ae_valid
        elif self.early_stopping_trigger == "acc":
            # return accuracy
            accuracy_valid = correct_predictions / num_predictions
            print(f"Accuracy from validation (classifier): {accuracy_valid}")
            return accuracy_valid, loss_ae_valid
    
    def validViT(self,epoch, valid_dataloader):
        """
            validation method used mainly for the Early stopping training
        """
        print (f"Validation for the epoch: {epoch} ...")
        
        # set temporary evaluation mode and empty cuda cache
        self.model.eval()
        
        T.cuda.empty_cache()
        
        # list of losses
        losses      = []
        
        # counters to compute accuracy
        correct_predictions = 0
        num_predictions = 0
        
        for (x,y) in tqdm(valid_dataloader):
            x = x.to(self.device)
            # y = y.to(self.device).to(T.float32)
            y = y.to(self.device).to(T.float32)
            
            with T.no_grad():
                out = self.model.forward(x) 
                logits      = out[0]
                
                # pred = self.sigmoid(logits)
                # classifier criterion
                if self.early_stopping_trigger == "loss":
                    # loss = self.bce(input=pred, target=y)   # pred bce version
                    loss = self.bce(input=logits, target=y)   # logits bce version
                    losses.append(loss.cpu().item())
                    
                elif self.early_stopping_trigger == "acc":
                    # prepare predictions and targets
                    pred = self.sigmoid(logits)
                    y_pred  = T.argmax(pred, -1).cpu().numpy()  # both are list of int (indices)
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
        
    def validAE(self,epoch, valid_dataloader):
        print (f"Validation for the epoch: {epoch} ...")
        
        # set temporary evaluation mode and empty cuda cache
        self.model.eval()
        self.autoencoder.eval()
        T.cuda.empty_cache()
        
        # list of losses

        losses_ae   = []
        
        for (x,y) in tqdm(valid_dataloader):
            
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
    
    # ------------------------------- training 
    
    # training wrapper
    def train(self, name_train, test_loop = False):
        # wrapper function to train
        if self.train_together:
            self.train_both(name_train, test_loop)
        else:
            name_folder_train = self.trainViT(name_train, test_loop)
            # self.trainAE(path_model_folder, path_results_folder, test_loop)
            self.trainAE(name_folder = name_folder_train, test_loop = test_loop)
    
    # train together classifier and autoencoder     
    @duration
    def train_both(self, name_train, test_loop = False):
        """
        Args:
            name_train (str) should include the scenario selected and the model name (i.e. ResNet50), keep this convention {scenario}_{model_name}
        """
        
        # set the full current model name 
        self.classifier_name = name_train
        
        # define the model dir path and create the directory
        current_date = date.today().strftime("%d-%m-%Y")    
        path_model_folder       = os.path.join(self.path_models,  name_train + "_v{}_".format(str(self.version)) + current_date)
        check_folder(path_model_folder, is_model= True)
        
        # define train dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # define valid dataloader
        valid_dataloader = DataLoader(self.valid_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # model in training mode
        self.model.train()
        self.autoencoder.train()
        
        # define the optimization algorithm
        self.optimizer          =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        self.optimizer_ae       =  Adam(self.autoencoder.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        
        # define the loss function and class weights, compute labels weights and cross entropy loss
        self.pos_weight_label = self.compute_class_weights(only_positive_weight=True)
        # self.bce       = T.nn.BCEWithLogitsLoss(weight=T.tensor(self.weights_labels)).to(device=self.device)
        self.bce       = T.nn.BCEWithLogitsLoss(pos_weight=T.tensor(self.pos_weight_label)).to(device=self.device)
        
        # learning rate scheduler
        if self.early_stopping_trigger == "loss":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr*0.01, verbose = True) # reduce of a half the learning rate 
        elif self.early_stopping_trigger == "acc":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr*0.01, verbose = True) # reduce of a half the learning rate 
        
        self.scheduler_ae = lr_scheduler.OneCycleLR(self.optimizer_ae, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        
        # define the gradient scaler to avoid weigths explosion
        scaler      = GradScaler()
        scaler_ae   = GradScaler()
        
        # initialize logger
        logger  = self.init_logger(path_model= path_model_folder, add2name="vit&ae")
        
        logger.write_model(self.autoencoder.getSummary(verbose=False), name_section = "AutoEncoder architecture")
        
        # intialize data structure to keep track of training performance
        loss_epochs     = []
        loss_epochs_ae  = []
        
        # initialzie the patience counter and history for early stopping
        valid_history       = []
        valid_ae_history    = []
        counter_stopping    = 0
        last_epoch          = 0
        
        # define best validation results
        if self.early_stopping_trigger == "loss":
            best_valid      = math.inf                # to minimize
        elif self.early_stopping_trigger == "acc":
            best_valid      = 0                       # to maximize
        best_valid_ae       = math.inf
        
        best_valid_epoch    = 0
        best_valid_ae_epoch = 0
        
        # initialize ditctionary best models
        best_model_dict     = copy.deepcopy(self.model.state_dict())
        best_model_ae_dict  = copy.deepcopy(self.autoencoder.state_dict())
        
        # learned epochs by the model initialization
        self.modelEpochs = 0
        
        tmp_losses      = []
        tmp_losses_ae   = []
        
        # T.autograd.set_detect_anomaly(True)
        # flag to detect forward problem with nan
        printed_nan = False
        
        # loop over epochs
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # define cumulative loss for the current epoch and max/min loss
            loss_epoch      = 0; max_loss_epoch     = 0; min_loss_epoch     = math.inf
            loss_epoch_ae   = 0; max_loss_epoch_ae  = 0; min_loss_epoch_ae  = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
            # list of max values for the attention maps over the epoch
            max_value_att_map = []
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= n_steps):
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(True)
                y = y.to(self.device)               # binary int encoding for each sample
                y = y.to(T.float)
                
                if T.isnan(x).any().item():
                    print("nan value in the input found")
                    continue
                
                # check there is any nan value in the input that causes instability
                
                # zeroing the gradient
                self.optimizer.zero_grad()
                
                # model forward and loss computation
                with autocast():   
                
                    out = self.model.forward(x) 
                    logits      = out[0]
                    att_maps    = out[2]
                    
                    #                                       apply activation function to logits
                    # pred = self.sigmoid(logits)
                    #                                       compute classification loss
                    loss      = self.bce(input=logits, target=y)   # bce from logits (no weights loaded)
                    # loss      = self.bce(input=pred, target=y)   # classic bce with "probabilities"

                # store max value attention maps
                max_value_att_map.append(T.max(att_maps).detach().cpu().item())
                
                if (T.isnan(loss).any().item() or T.isinf(loss).any().item())and not(printed_nan):
                    print(loss)
                    print(logits)
                    print(T.norm(logits))
                    print(y)
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            print(f'Parameter: {name}, Gradient Magnitude: {param.grad.norm().item()}')
                    
                    printed_nan = True
                    break  #
                    
                
                loss_value = loss.cpu().item()
                tmp_losses.append(loss_value)
                
                print_every = 50
                if (step_idx+1)%print_every == 1:
                    # print("norm logits every 100 epochs ->", T.norm(logits).cpu().item())
                    print(f"avg loss every {print_every} epochs ->", sum(tmp_losses)/len(tmp_losses))
                    tmp_losses = []
                
                if loss_value>max_loss_epoch    : max_loss_epoch = round(loss_value,4)
                if loss_value<min_loss_epoch    : min_loss_epoch = round(loss_value,4)
                
                # update total loss    
                loss_epoch += loss_value   # from tensor with single value to int and accumulation
                
                # loss backpropagation
                scaler.scale(loss).backward()
                
                # compute updates using optimizer
                scaler.step(self.optimizer)

                # update weights through scaler
                scaler.update()
                
                # (optional) gradient clipping 
                # T.nn.utils.clip_grad_norm_(self.model.parameters(), 10) 
                    
                #                                      compute reconstruction loss
                self.optimizer_ae.zero_grad()
                
                x_ae = att_maps.detach().clone()
                x_ae.requires_grad_(True)
                x_ae.to(device=self.device)
                
                with autocast():
                    rec_att_maps = self.autoencoder(x_ae)

                    loss_ae      = self.mae(rec_att_maps, x_ae)
                
                loss_value_ae = loss_ae.cpu().item()
                
                tmp_losses_ae.append(loss_value_ae)
                
                if (step_idx+1)%print_every == 1:
                    print(f"avg ae loss every {print_every} epochs ->", sum(tmp_losses_ae)/len(tmp_losses_ae))
                    print(f"max att map: {T.max(att_maps)}", f"min att map: {T.min(att_maps)}", f"avg sum att map: {T.mean(T.sum(att_maps.view(att_maps.shape[0], -1), dim = 0))}")
                    tmp_losses_ae = []
                
                if loss_value_ae>max_loss_epoch_ae    : max_loss_epoch_ae = round(loss_value_ae,4)
                if loss_value_ae<min_loss_epoch_ae    : min_loss_epoch_ae = round(loss_value_ae,4)
                
                # update total loss    
                loss_epoch_ae += loss_value_ae   # from tensor with single value to int and accumulation
            
                scaler_ae.scale(loss_ae).backward()                     # loss_ae.backward() # self.optimizer_ae.step()
                
                scaler_ae.step(self.optimizer_ae)
                
                scaler_ae.update()
                
                self.scheduler_ae.step()
                
                # (optional) gradient clipping 
                # T.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 10) 
                    
            if printed_nan: break
                     
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            avg_max_attention = sum(max_value_att_map)/len(max_value_att_map)   
            print("Average max attention value: {}".format(avg_max_attention))
            
            avg_loss_ae = round(loss_epoch_ae/n_steps,4)
            loss_epochs_ae.append(avg_loss_ae)
            print("Average loss autoencoder: {}".format(avg_loss_ae))
            
            # include validation here if needed
            criterion, criterion_ae = self.valid(epoch=epoch_idx+1, valid_dataloader= valid_dataloader)
            valid_history.append(criterion)
            valid_ae_history.append(criterion_ae)  
            
            # look for best models *.*
            
            if self.early_stopping_trigger  == "loss":              # ViT
                if criterion < best_valid:
                    best_valid = criterion
                    best_model_dict = copy.deepcopy(self.model.state_dict())
                    best_valid_epoch = epoch_idx+1
                    print("** new best classifier **")
            elif self.early_stopping_trigger == "acc":
                if criterion > best_valid:
                    best_valid = criterion
                    best_model_dict = copy.deepcopy(self.model.state_dict())
                    best_valid_epoch = epoch_idx+1
                    print("** new best classifier **") 
            
            if criterion_ae < best_valid_ae:                        # AE 
                best_valid_ae = criterion_ae
                best_model_ae_dict = copy.deepcopy(self.autoencoder.state_dict())
                best_valid_ae_epoch = epoch_idx +1
                print("** new best autoencoder **")
        
            # initialize not early stopping
            early_exit = False 
            
            # early stopping update
            if epoch_idx > 0 and last_epoch >= self.start_early_stopping:
                print("Early stopping step ...")
                
                if self.early_stopping_trigger == "loss":
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
                            
                elif self.early_stopping_trigger == "acc":
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
                          "min_loss": min_loss_epoch, self.early_stopping_trigger + "_valid": criterion,
                         }
            
            epoch_data_ae = {
                            "epoch": last_epoch, "avg_loss_ae": avg_loss_ae, "max_loss_ae": max_loss_epoch_ae,
                            "min_loss_ae": min_loss_epoch_ae, "loss_valid": criterion_ae
            }
            
            logger.log(epoch_data)
            logger.log(epoch_data_ae)
            
            # test epochs loop for debug   
            if test_loop and last_epoch == 5: break
            
            # exit for early stopping if is the case
            if early_exit: break 
            
            # lr scheduler step based on validation result
            self.scheduler.step(criterion)
            

        # log GPU memory statistics during training
        logger.log_mem(T.cuda.memory_summary(device=self.device))
        
        # create path for the model save
        # classifier
        name_model_file         = str(last_epoch) +'.ckpt'
        name_best_model_file    = str(best_valid_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        path_best_model_save    = os.path.join(path_model_folder, name_best_model_file)
        # autoencoder
        name_model_ae_file      = "ae_" + str(last_epoch) +'.ckpt'
        name_best_model_ae_file = "ae_" + str(best_valid_ae_epoch) +'.ckpt'
        path_model_ae_save      = os.path.join(path_model_folder, name_model_ae_file)
        path_best_model_ae_save = os.path.join(path_model_folder, name_best_model_ae_file)
        
        # create path for the model results
        path_results_folder     = os.path.join(self.path_results, name_train + "_v{}_".format(str(self.version)) + current_date)
        check_folder(path_results_folder) # create if doesn't exist
        # classifier
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        name_valid_file         = "valid_{}_{}.png".format(self.early_stopping_trigger, str(last_epoch))
        # autoencoder
        name_loss_ae_file          = 'loss_ae_'+ str(last_epoch) +'.png'
        name_valid_ae_file         = "valid_ae_loss_{}.png".format(str(last_epoch))
        
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs        = last_epoch
        
        # save loss plot
        if test_loop:
            # main model
            plot_loss(loss_epochs, title_plot= "classifier", path_save = None)
            plot_valid(valid_history, title_plot= "classifier  "+ self.early_stopping_trigger , path_save = None)
            
            plot_loss(loss_epoch_ae, title_plot = "AE", path_save = None)
            plot_valid(valid_ae_history, title_plot  = "AE loss", path_save = None)
            
        else:
            # main model
            plot_loss(loss_epochs, title_plot= "classifier", path_save = os.path.join(path_results_folder, name_loss_file))
            plot_loss(loss_epochs, title_plot= "classifier", path_save = os.path.join(path_model_folder,name_loss_file), show=False)
            plot_valid(valid_history, title_plot= "classifier "+ self.early_stopping_trigger, path_save = os.path.join(path_results_folder, name_valid_file))
            plot_valid(valid_history, title_plot= "classifier "+ self.early_stopping_trigger, path_save = os.path.join(path_model_folder,name_valid_file), show=False)
            
            # autoencoder
            plot_loss(loss_epochs_ae, title_plot= "AE", path_save = os.path.join(path_results_folder, name_loss_ae_file))
            plot_loss(loss_epochs_ae, title_plot= "AE", path_save = os.path.join(path_model_folder,name_loss_ae_file), show=False)
            plot_valid(valid_ae_history, title_plot= "AE loss", path_save = os.path.join(path_results_folder, name_valid_ae_file))
            plot_valid(valid_ae_history, title_plot= "AE loss", path_save = os.path.join(path_model_folder,name_valid_ae_file), show=False)
            
        #                   save models
        # save bests
        saveModel(best_model_dict, path_best_model_save, is_dict= True)
        saveModel(best_model_ae_dict, path_best_model_ae_save, is_dict= True)
        
        # save latest
        saveModel(self.model, path_model_save)
        saveModel(self.autoencoder, path_model_ae_save)
        
        # terminate the log session
        logger.end_log(model_results_folder_path=path_results_folder)
        
        self.autoencoder.eval()
        self.model.eval()

    # train separately
    @duration
    def trainViT(self, name_train, test_loop = False):
        """
        Args:
            name_train (str) should include the scenario selected and the model name (i.e. ResNet50), keep this convention {scenario}_{model_name}
        """
        
        # set the full current model name 
        self.classifier_name = name_train
        
        # define the model dir path and create the directory
        current_date = date.today().strftime("%d-%m-%Y")    
        name_folder = name_train + "_v{}_".format(str(self.version)) + current_date
    
        path_model_folder       = os.path.join(self.path_models,  name_folder)
        self.path_model_folder  = path_model_folder
        check_folder(path_model_folder, is_model= True)
        # create path for the model results
        path_results_folder     = os.path.join(self.path_results, name_folder)
        self.path_results_folder = path_results_folder
        check_folder(path_results_folder) # create if doesn't exist
        
        # define train dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # define valid dataloader
        valid_dataloader = DataLoader(self.valid_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # model in training mode
        self.model.train()
        
        # define the optimization algorithm
        self.optimizer          =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        # define the loss function and class weights, compute labels weights and cross entropy loss
        self.pos_weight_label = self.compute_class_weights(only_positive_weight=True)
        
        self.bce       = T.nn.BCEWithLogitsLoss(pos_weight=T.tensor(self.pos_weight_label)).to(device=self.device)
        
        # learning rate schedulers
        if self.early_stopping_trigger == "loss":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr*0.01, verbose = True) # reduce of a half the learning rate 
        elif self.early_stopping_trigger == "acc":
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor = 0.5, patience = 5, cooldown = 2, min_lr = self.lr*0.01, verbose = True) # reduce of a half the learning rate 
        
        # define the gradient scaler to avoid weigths explosion
        scaler      = GradScaler()
        
        # initialize logger
        logger  = self.init_logger(path_model= path_model_folder, add2name="vit")
        
        # intialize data structure to keep track of training performance
        loss_epochs     = []
        
        # initialzie the patience counter and history for early stopping
        valid_history       = []
        counter_stopping    = 0
        last_epoch          = 0
        
        # define best validation results
        if self.early_stopping_trigger == "loss":
            best_valid = math.inf                # to minimize
        elif self.early_stopping_trigger == "acc":
            best_valid = 0                       # to maximize
        best_valid_epoch = 0

        # initialize ditctionary best models
        best_model_dict = copy.deepcopy(self.model.state_dict())

        # learned epochs by the model initialization
        self.modelEpochs = 0
        tmp_losses      = []
        
        # T.autograd.set_detect_anomaly(True)
        # flag to detect forward problem with nan
        printed_nan = False
        
        # loop over epochs
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # define cumulative loss for the current epoch and max/min loss
            loss_epoch      = 0; max_loss_epoch     = 0; min_loss_epoch     = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
            
            max_value_att_map = []
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= n_steps):
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(True)
                y = y.to(self.device)               # binary int encoding for each sample
                y = y.to(T.float)
                
                if T.isnan(x).any().item():
                    print("nan value in the input found")
                    continue
                
                # check there is any nan value in the input that causes instability
                # zeroing the gradient
                self.optimizer.zero_grad()
                # model forward and loss computation
                with autocast():   

                    out = self.model.forward(x) 
                    logits      = out[0]
                    att_maps    = out[2]
                    
                    
                    # apply activation function to logits
                    # pred = self.sigmoid(logits)
                        
                    #                                       compute classification loss
                    loss      = self.bce(input=logits, target=y)   # bce from logits (no weights loaded)
                    # loss      = self.bce(input=pred, target=y)   # classic bce with "probabilities"

                
                max_value_att_map.append(T.max(att_maps).detach().cpu().item())
                
                if (T.isnan(loss).any().item() or T.isinf(loss).any().item())and not(printed_nan):
                    print(loss)
                    print(logits)
                    print(T.norm(logits))
                    print(y)
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            print(f'Parameter: {name}, Gradient Magnitude: {param.grad.norm().item()}')
                    
                    printed_nan = True
                    break  #
                    
                
                loss_value = loss.cpu().item()
                tmp_losses.append(loss_value)
                
                print_every = 50
                if (step_idx+1)%print_every == 0:
                    # print("norm logits every 100 epochs ->", T.norm(logits).cpu().item())
                    print(f"avg loss every {print_every} epochs ->", sum(tmp_losses)/len(tmp_losses))
                    print(f"max att map: {T.max(att_maps)}", f"min att map: {T.min(att_maps)}", f"avg sum att map: {T.mean(T.sum(att_maps.view(att_maps.shape[0], -1), dim = 0))}")
                    tmp_losses = []
                
                if loss_value>max_loss_epoch    : max_loss_epoch = round(loss_value,4)
                if loss_value<min_loss_epoch    : min_loss_epoch = round(loss_value,4)
                
                # update total loss    
                loss_epoch += loss_value   # from tensor with single value to int and accumulation
                
                # loss backpropagation
                scaler.scale(loss).backward()
                
                # compute updates using optimizer
                scaler.step(self.optimizer)

                # update weights through scaler
                scaler.update()
                
                # (optional) gradient clipping 
                # T.nn.utils.clip_grad_norm_(self.model.parameters(), 10) 
                         
            if printed_nan: break
                     
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            avg_max_attention = sum(max_value_att_map)/len(max_value_att_map)   
            print("Average max attention value: {}".format(avg_max_attention))
            
            # include validation here if needed
            criterion = self.validViT(epoch=epoch_idx+1, valid_dataloader= valid_dataloader)
            valid_history.append(criterion)
            
            # look for best models *.*
            
            if self.early_stopping_trigger  == "loss":
                if criterion < best_valid:
                    best_valid = criterion
                    best_model_dict = copy.deepcopy(self.model.state_dict())
                    best_valid_epoch = epoch_idx+1
                    print("** new best classifier **")
            elif self.early_stopping_trigger == "acc":
                if criterion > best_valid:
                    best_valid = criterion
                    best_model_dict = copy.deepcopy(self.model.state_dict())
                    best_valid_epoch = epoch_idx+1
                    print("** new best classifier **")   
            
            # initialize not early stopping
            early_exit = False 
            
            # early stopping update
            if epoch_idx > 0 and last_epoch >= self.start_early_stopping:
                print("Early stopping step ...")
                
                if self.early_stopping_trigger == "loss":
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
                            
                elif self.early_stopping_trigger == "acc":
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
                          "min_loss": min_loss_epoch, self.early_stopping_trigger + "_valid": criterion,
                         }
            
            logger.log(epoch_data)
            
            # test epochs loop for debug   
            if test_loop and last_epoch == 5: break
            
            # exit for early stopping if is the case
            if early_exit: break 
            
            # lr scheduler step based on validation result
            self.scheduler.step(criterion)
            

        # log GPU memory statistics during training
        logger.log_mem(T.cuda.memory_summary(device=self.device))
        
        # create path for the model save and loss plot in results
        # classifier
        name_model_file         = str(last_epoch) +'.ckpt'
        name_best_model_file    = str(best_valid_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        path_best_model_save    = os.path.join(path_model_folder, name_best_model_file)

        # classifier
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        name_valid_file         = "valid_{}_{}.png".format(self.early_stopping_trigger, str(last_epoch))
        
        
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs        = last_epoch
        
        # save loss plot

        # main model
        plot_loss(loss_epochs, title_plot= "classifier", path_save = os.path.join(path_results_folder, name_loss_file))
        plot_loss(loss_epochs, title_plot= "classifier", path_save = os.path.join(path_model_folder,name_loss_file), show=False)
        plot_valid(valid_history, title_plot= "classifier "+ self.early_stopping_trigger, path_save = os.path.join(path_results_folder, name_valid_file))
        plot_valid(valid_history, title_plot= "classifier "+ self.early_stopping_trigger, path_save = os.path.join(path_model_folder,name_valid_file), show=False)

        # save model
        saveModel(self.model, path_model_save)
        saveModel(best_model_dict, path_best_model_save, is_dict= True)
        
        # terminate the log session
        logger.end_log(model_results_folder_path=path_results_folder)
        
        self.model.eval()
        
        return name_folder
    
    @duration
    def trainAE(self, name_folder, test_loop = False):
        """
        Args:
            name_train (str) should include the scenario selected and the model name (i.e. ResNet50), keep this convention {scenario}_{model_name}
        """
        print(f"AE training for model {name_folder}")
        
        # define paths for saving models
        path_model_folder       = os.path.join(self.path_models,  name_folder)
        self.path_model_folder  = path_model_folder

        path_results_folder     = os.path.join(self.path_results, name_folder)
        self.path_results_folder = path_results_folder

        # define train dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)

        # define valid dataloader
        valid_dataloader = DataLoader(self.valid_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)

        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))

        # model in training mode
        self.autoencoder.train()

        self.optimizer_ae       =  Adam(self.autoencoder.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        
        self.scheduler_ae = lr_scheduler.OneCycleLR(self.optimizer_ae, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs_AE, pct_start=0.3)
       
        # define the gradient scaler to avoid weigths explosion
        scaler_ae   = GradScaler()

        # initialize logger
        logger  = self.init_logger_ae(path_model= path_model_folder, add2name="ae")
        
        # intialize data structure to keep track of training performance
        loss_epochs_ae  = []

        # initialzie the patience counter and history for early stopping
        valid_ae_history    = []
        last_epoch          = 0

        # learned epochs by the model initialization
        self.modelEpochs_ae = 0

        # T.autograd.set_detect_anomaly(True)

        # loop over epochs
        for epoch_idx in range(self.n_epochs_AE):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # define cumulative loss for the current epoch and max/min loss
            loss_epoch_ae   = 0; max_loss_epoch_ae  = 0; min_loss_epoch_ae  = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
                
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= n_steps):
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(False)
                
                if T.isnan(x).any().item():
                    print("nan value in the input found")
                    continue
                
                # check there is any nan value in the input that causes instability
                
                # model forward and loss computation
                with T.no_grad():
                    
                    # _, _, att_maps  = self.model.forward(x) 
                    out = self.model.forward(x) 
                    att_maps    = out[2]
                
                
                #                                      compute reconstruction loss
                self.optimizer_ae.zero_grad()
                
                x_ae = att_maps.detach().clone()
                x_ae.requires_grad_(True)
                x_ae.to(device=self.device)

                
                if self.model_ae_type == "vae":
                    rec_att_map, mean, logvar = self.autoencoder.forward(x_ae, train=True)
                    loss_ae     = self.autoencoder.loss_function(rec_att_map, x_ae, mean, logvar, rec_loss="mae", kld_loss="sum")
                else: 
                    with autocast():
                        rec_att_map = self.autoencoder(x_ae)
                        loss_ae     = self.mae(rec_att_map, x_ae)
                
                loss_value_ae = loss_ae.cpu().item()
                
                if loss_value_ae>max_loss_epoch_ae    : max_loss_epoch_ae = round(loss_value_ae,4)
                if loss_value_ae<min_loss_epoch_ae    : min_loss_epoch_ae = round(loss_value_ae,4)
                
                # update total loss    
                loss_epoch_ae += loss_value_ae   # from tensor with single value to int and accumulation
            
                scaler_ae.scale(loss_ae).backward()                     # loss_ae.backward() # self.optimizer_ae.step()
                
                scaler_ae.step(self.optimizer_ae)
                
                scaler_ae.update()
                
                self.scheduler_ae.step()
            
                # (optional) gradient clipping 
                # T.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 10) 
                           
            # compute average loss for the epoch
            avg_loss_ae = round(loss_epoch_ae/n_steps,4)
            loss_epochs_ae.append(avg_loss_ae)
            print("Average loss autoencoder: {}".format(avg_loss_ae))
            
            # include validation here if needed
            criterion_ae = self.validAE(epoch=epoch_idx+1, valid_dataloader= valid_dataloader)
            valid_ae_history.append(criterion_ae)  
            
            
            # create dictionary with info frome epoch: loss + valid, and log it
            epoch_data_ae = {
                            "epoch": last_epoch, "avg_loss_ae": avg_loss_ae, "max_loss_ae": max_loss_epoch_ae,
                            "min_loss_ae": min_loss_epoch_ae, "loss_valid": criterion_ae
            }
            
            logger.log(epoch_data_ae)
            
            # test epochs loop for debug   
            if test_loop and last_epoch == 5: break
            

        # log GPU memory statistics during training
        logger.log_mem(T.cuda.memory_summary(device=self.device))

        # create path for the model save
        
        # autoencoder
        name_model_ae_file      = "ae_" + str(last_epoch) +'.ckpt'
        path_model_ae_save      = os.path.join(path_model_folder, name_model_ae_file)

        # autoencoder
        name_loss_ae_file          = 'loss_ae_'+ str(last_epoch) +'.png'
        name_valid_ae_file         = "valid_ae_loss_{}.png".format(str(last_epoch))

        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs_ae        = last_epoch

        # save loss plot
        # autoencoder
        plot_loss(loss_epochs_ae, title_plot= "AE", path_save = os.path.join(path_results_folder, name_loss_ae_file))
        plot_loss(loss_epochs_ae, title_plot= "AE", path_save = os.path.join(path_model_folder,name_loss_ae_file), show=False)
        plot_valid(valid_ae_history, title_plot= "AE loss", path_save = os.path.join(path_results_folder, name_valid_ae_file))
        plot_valid(valid_ae_history, title_plot= "AE loss", path_save = os.path.join(path_model_folder,name_valid_ae_file), show=False)
            
        # save models
        saveModel(self.autoencoder, path_model_ae_save)

        # terminate the log session
        logger.end_log(model_results_folder_path=path_results_folder)

        self.autoencoder.eval()
    
    # ------------------------------- testing, load & forward 
    
    def test(self):
        # call same test function but don't load again the data
        super().test(load_data=False)
    
    # load specific functions
    
    def load_both(self, folder_model, epoch, epoch_ae = None):
        """ load both ViT model and autoencoder"""
        # load main model
        super().load(folder_model, epoch)
        
        if epoch_ae == None:
            epoch_ae = self.n_epochs_AE
        # load autoencoder
        try:
            self.path2model_ae         = os.path.join(self.path_models,  folder_model,"ae_"+ str(epoch_ae) + ".ckpt")
            loadModel(self.autoencoder, self.path2model_ae)
            self.autoencoder.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except Exception as e:
            print(e)
            print("No model: {} found for the epoch: {} in the folder: {}".format(folder_model, epoch, self.path_models))
    
    def load_ae(self, folder_model, epoch_ae):
        try:
            self.path2model_ae         = os.path.join(self.path_models,  folder_model,"ae_"+ str(epoch_ae) + ".ckpt")
            loadModel(self.autoencoder, self.path2model_ae)
            self.autoencoder.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except Exception as e:
            print(e)
            print("No model: {} found for the epoch: {} in the folder: {}".format(folder_model, epoch_ae, self.path_models))
    
    def test_ae(self):
        """
            Void function that computes test on the autoencoder

            Params:
            load_data (boolean): parameter used to specify if is required to load data or has been already done, Default is True
            
            Returns:
            None 
        """    
        
        # define test dataloader
        test_dataloader = DataLoader(self.test_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)

        # compute number of batches for test
        n_steps = len(test_dataloader)
        print("Number of batches for test: {}".format(n_steps))
        
        # model in evaluation mode
        self.model.eval()  
        
        # define the array to store the result
        losses_mae              = np.empty((0), dtype= np.float32)
        losses_mse              = np.empty((0), dtype= np.float32)
        
        T.cuda.empty_cache()
        
        # loop over steps
        for _,(x,y) in tqdm(enumerate(test_dataloader), total= n_steps):
            
            # prepare samples/targets batches 
            x = x.to(self.device)
                    
            with T.no_grad():
                
                out = self.model.forward(x) 
                att_maps    = out[2]

                #                                      compute reconstruction loss
                
                x_ae = att_maps.detach().clone()
                x_ae.requires_grad_(True)
                x_ae.to(device=self.device)

                
                # if self.model_ae_type == "vae":
                #     rec_att_map, _, _  = self.autoencoder(x_ae)
                # else: 
                rec_att_map = self.autoencoder(x_ae)
                    
                mae_loss     = np.expand_dims(self.mae(rec_att_map, x_ae).cpu().item(), axis = 0)
                mse_loss     = np.expand_dims(self.mse(rec_att_map, x_ae).cpu().item(), axis = 0)
                
        
            losses_mae  = np.append(losses_mae, mae_loss, axis  =0)
            losses_mse  = np.append(losses_mse, mse_loss, axis  =0)
            
              
        # compute metrics 
        
        mae_metric = float(np.mean(losses_mae, axis=0))
        mse_metric = float(np.mean(losses_mse, axis=0))
        
        
        metrics = {"MAE": mae_metric, "MSE": mse_metric}
        
        print("Reconstruction error from test:\n", metrics)
        
        saveJson(os.path.join(self.path_models, self.classifier_name, "metrics_ae.json"), metrics)
        
        # create folder for the results
        # check_folder(self.path2model_results)
            
        # compute metrics from test data
        # metrics_binClass(predictions, targets, predicted_probabilities, epoch_model= str(self.modelEpochs), path_save = self.path2model_results)
    
    # Override of superclass forward method
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
        
        out = self.model.forward(x) 
        logits = out[0]
        
        probs       = self.sigmoid(logits)    # change to softmax in multi-class context
        pred        = T.argmax(probs, -1)
        fake_prob   = probs[:,1]   # positive class probability (fake probability)
        
        return pred, fake_prob, logits

if __name__ == "__main__":
    #                           [Start test section] 
    scenario_prog       = 1
    data_scenario       = None
    
    if scenario_prog == 0: 
        data_scenario = "content"
    elif scenario_prog == 1:
        data_scenario = "group"
    elif scenario_prog == 2:
        data_scenario = "mix"
    
    scenario_setting = getScenarioSetting()[data_scenario]    # type of content/group
    
    # ________________________________ v6  ________________________________
    
    def train_v6_scenario(model_type, add_name ="", patch_size = None, emb_size = None):
        
        bin_classifier = DFD_BinViTClassifier_v6(scenario = data_scenario, patch_size= patch_size, emb_size= emb_size,
                                                 useGPU= True, model_type=model_type)
        if add_name != "":
            bin_classifier.train(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = False)
        else:
            bin_classifier.train(name_train= scenario_setting + "_" + model_type, test_loop = False)

    def test_v6_metrics(name_model, epoch, model_type, patch_size = None, emb_size = None):
        bin_classifier = DFD_BinViTClassifier_v6(scenario = data_scenario, patch_size= patch_size, emb_size= emb_size,
                                                 useGPU= True, model_type= model_type)
        bin_classifier.load(name_model, epoch)
        bin_classifier.test()
        
    def train_test_v6_metrics(model_type, add_name ="", patch_size = None, emb_size = None):
        bin_classifier = DFD_BinViTClassifier_v6(scenario = data_scenario, patch_size= patch_size, emb_size= emb_size,
                                                 useGPU= True, model_type=model_type)
        if add_name != "":
            bin_classifier.train_and_test(name_train= scenario_setting + "_" + model_type + "_" + add_name)
        else:
            bin_classifier.train_and_test(name_train= scenario_setting + "_" + model_type)
            
    # ________________________________ v7  ________________________________
    
    # ------- attention map test
    
    def test_generic_attention_map():

        from PIL import Image
        import matplotlib.pyplot as plt
        from timm.models import create_model
        from torchvision import transforms
        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

        def to_tensor(img):
            transform_fn = Compose([Resize(249, 3), CenterCrop(224), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            return transform_fn(img)

        def show_img(img):
            img = np.asarray(img)
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.axis('off')
            plt.show()

        def show_img2(img1, img2, alpha=0.8):
            img1 = np.asarray(img1)
            img2 = np.asarray(img2)
            plt.figure(figsize=(10, 10))
            plt.imshow(img1)
            plt.imshow(img2, alpha=alpha)
            plt.axis('off')
            plt.show()

        def my_forward_wrapper(attn_obj):
            def my_forward(x):
                B, N, C = x.shape
                print("B: ", B)
                print("N: ", N)
                print("C: ", C)
                
                qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
                
                print("qkv shape", qkv.shape)
                
                q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
                
                print("q shape", q.shape)
                print("k shape", k.shape)
                print("v shape", v.shape)
                

                attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
                attn = attn.softmax(dim=-1)
                attn = attn_obj.attn_drop(attn)
                
                # print(attn.shape)
                
                attn_obj.attn_map = attn
                # attn_obj.cls_attn_map = attn[:, :, 0, 2:]
                attn_obj.cls_attn_map = attn[:, :, 0, 1:]

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = attn_obj.proj(x)
                x = attn_obj.proj_drop(x)
                return x
            return my_forward
        
        bin_classifier = DFD_BinViTClassifier_v7(scenario = "content", useGPU= True)
        data_iter = bin_classifier.train_dataset
        x1, y = data_iter.__getitem__(0)
        x2, _ = data_iter.__getitem__(1)
        

        
        # model = create_model('deit_small_distilled_patch16_224', pretrained=True)
        model = create_model(model_name='vit_base_patch16_224.augreg_in21k' , pretrained=True, num_classes=2, drop_rate=0.1)
        
        # bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type)
        # bin_classifier.load(name_model, epoch)
        
        
        model.blocks[-1].attn.forward = my_forward_wrapper(model.blocks[-1].attn)
        # model = bin_classifier.model.model_vit

        x = T.stack((x1,x2))
        print("image batch shape: ", x.shape)
        
        # x = x.unsqueeze(0)

        _ = model(x)
        
        att_map = model.blocks[-1].attn.attn_map.mean(dim=1).detach()
        att_map = att_map[:, 1:, 1:].view(-1,1,196,196)
        print("full attention map: ", att_map.shape)
        
        # print(model.blocks[-1].attn.cls_attn_map.mean(dim=1).shape)
        att_map_cls = model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(-1, 14, 14).detach()
        att_map_cls = att_map_cls.unsqueeze(dim = 1)
        
        print("attention map cls (batch) shape: ", att_map_cls.shape)
        
        # print(model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(-1,14,14).shape)
        # cls_weight = model.blocks[-1].attn.cls_attn_map[:].mean(dim=1).view(14, 14).detach()
        # print(cls_weight.shape)
        
        
        print("att_map max: ", T.max(att_map))
        print("att_map min: ", T.min(att_map))
        print("att_map cls max: ", T.max(att_map_cls))
        print("att_map cls min: ", T.min(att_map_cls))
        
        att_map_up          = F.interpolate(att_map, (224,224), mode = "bilinear")
        att_map_cls_up      = F.interpolate(att_map_cls, (224, 224), mode='bilinear')
        
        

        x_show          = (x[0].permute(1, 2, 0) + 1)/2
        
        att_map         = att_map[0].permute(1,2,0)
        att_map_up      = att_map_up[0].permute(1,2,0)
        att_map_cls     = att_map_cls[0].permute(1,2,0)
        att_map_cls_up  = att_map_cls_up[0].permute(1,2,0)
        
        
        show_img(x_show)
        
        
        show_img(att_map)
        show_img(att_map_up)
        
        show_img(att_map_cls)
        show_img(att_map_cls_up)
        
        print(x_show.shape)
        print(att_map_cls_up.shape)
        
        show_img2(x_show, att_map_cls_up, alpha=0.8)
    
    def test_attention_map_v7(name_model, epoch, epoch_ae, prog_model = 3, model_type = "ViTEA_timm", autoencoder_type = "vae"):
        
        bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type,\
                                                 transform_prog=0, prog_pretrained_model=prog_model, model_ae_type= autoencoder_type)
        
        bin_classifier.load_both(name_model, epoch, epoch_ae= epoch_ae)
        data_iter = bin_classifier.test_dataset
        save    = True
        img_id  = 0
        
        path_save_images = os.path.join(bin_classifier.path_models, bin_classifier.classifier_name)
        
        print(f"path save for images: {path_save_images}")
        
        img, y = data_iter.__getitem__(img_id)
        
        # img = trans_input_base()(Image.open("./static/test_image_attention.png"))
        
        showImage(img, save_image= save, name="attention_original_" + str(img_id), path_save=path_save_images)
    
        img = img.unsqueeze(dim=0)
    
        img = img.to(device = bin_classifier.device)
        
        out = bin_classifier.model.forward(img)
        
        att_map         = out[2]
        att_map_patches = out[3]
        
        print("att_map cls: shape, max, min: ", att_map.shape, T.max(att_map), T.min(att_map))
        
        showImage(att_map[0], has_color= False, save_image= save, name="attention_map_" + str(img_id), path_save=path_save_images)
        
        print("att_map patches: shape, max, min: ", att_map_patches.shape, T.max(att_map_patches), T.min(att_map_patches))
        
        showImage(att_map_patches[0], has_color= False, save_image= save, name="attention_map_patches_" + str(img_id), path_save=path_save_images)
        
        blended_results, _  = include_attention(img, att_map, alpha= 0.7)
        
        showImage(blended_results[0], save_image= save, name="attention_fused_" + str(img_id), path_save=path_save_images)
        
        show_imgs_blend(img[0].cpu(), att_map[0].cpu(), alpha=0.8, save_image= save, name="attention_blend_" + str(img_id), path_save=path_save_images)
        
        # reconstruction 
        # if bin_classifier.model_ae_type == "vae":
        #     rec_att_map, _, _  = bin_classifier.autoencoder.forward(att_map)
        # else:    
        rec_att_map = bin_classifier.autoencoder.forward(att_map)
                
        print("rec_att_map: shape, max, min: ", rec_att_map.shape, T.max(rec_att_map), T.min(rec_att_map))

        showImage(rec_att_map[0], has_color= False, save_image= save, name="attention_map_AE_" + str(img_id), path_save=path_save_images)
        
    
    # ------- train
    
    def train_v7_scenario(prog_vit, model_type = "ViTEA_timm", add_name ="", autoencoder_type = "vae"):
        
        bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type=model_type,\
                                                prog_pretrained_model= prog_vit, model_ae_type= autoencoder_type)
        if add_name != "":
            bin_classifier.train(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = False)
        else:
            bin_classifier.train(name_train= scenario_setting + "_" + model_type, test_loop = False)

    def train_v7_scenario_separately(prog_vit, model_type = "ViTEA_timm", add_name ="", autoencoder_type = "vae", test_loop = False):
        
        bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type=model_type,\
                                                 train_together=False, prog_pretrained_model= prog_vit, model_ae_type= autoencoder_type)
        
        if add_name != "":
            bin_classifier.train(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = test_loop)
        else:
            bin_classifier.train(name_train= scenario_setting + "_" + model_type, test_loop = test_loop)

    def trainViT_v7_scenario(prog_vit, model_type = "ViTEA_timm", add_name ="", test_loop = False):
        bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type=model_type,\
                                            train_together=False, prog_pretrained_model= prog_vit)
        
        if add_name != "":
            bin_classifier.trainViT(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = test_loop)
        else:
            bin_classifier.trainViT(name_train= scenario_setting + "_" + model_type, test_loop = test_loop)
        
    def trainAE_v7_scenario(name_folder_train, epoch_vit, prog_vit, model_type = "ViTEA_timm", autoencoder_type = "vae", test_loop = False):
            bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type=model_type,\
                                        train_together=False, prog_pretrained_model= prog_vit, model_ae_type= autoencoder_type)
            
            bin_classifier.load(name_folder_train, epoch_vit)
            
            bin_classifier.trainAE(name_folder=name_folder_train, test_loop=test_loop)
        
    
    
    # ------- test
     
    def test_v7_metrics(name_folder_train, epoch, prog_model = 3, model_type = "ViTEA_timm"):
        bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type, prog_pretrained_model= prog_model, train_together= False)
        bin_classifier.load(name_folder_train, epoch)
        bin_classifier.test()
    
    def test_v7_ae_metrics(name_folder_train, epoch, epoch_ae, prog_model = 3, model_type = "ViTEA_timm", autoencoder_type = "vae"):
        bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type, model_ae_type=autoencoder_type, 
                                                 prog_pretrained_model= prog_model, train_together= False)
        
        bin_classifier.load_both(name_folder_train,epoch=epoch, epoch_ae= epoch_ae)
        
        bin_classifier.test_ae()
    
    def test_v7_both_metrics(name_folder_train, epoch, epoch_ae, prog_model  = 3, model_type = "ViTEA_timm", autoencoder_type = "vae"):
        bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type, model_ae_type=autoencoder_type, 
                                            prog_pretrained_model= prog_model, train_together= False)
        
        bin_classifier.load_both(name_folder_train,epoch=epoch, epoch_ae= epoch_ae)
        
        bin_classifier.test()
        bin_classifier.test_ae()
   
    
    # test_attention_map_v7("faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_13-02-2024", epoch=25, epoch_ae=25, prog_model=3, autoencoder_type = "vae")
    
    # TODO train gan, not good results with tiny DeiT
    # train_v7_scenario_separately(prog_vit=3, add_name="DeiT_tiny_separateTrain")
    # test_v7_both_metrics("gan_ViTEA_timm_DeiT_tiny_separateTrain_v7_20-02-2024", 54, 75)
    
    
    #                           [End test section] 
    """ 
            Past test/train launched: 
    # faces:
        #                                           v6
        train_v6_scenario(model_type="ViT_B16_pretrained")              #224
        train_v6_scenario(model_type="ViT_B16_pretrained")              #224 
        train_v6_scenario(model_type="ViT_base_S32")                    #224
        train_v6_scenario(model_type="ViT_base_S16", patch_size= 16)    #224
        train_v6_scenario(model_type="ViT_base_S32", add_name="training+") #224


        
        test_v6_metrics(name_model = "faces_ViT_B16_pretrained_v6_27-01-2024", epoch = 9, model_type="ViT_B16_pretrained")
        test_v6_metrics(name_model = "faces_ViT_B16_pretrained_v6_29-01-2024", epoch = 15, model_type="ViT_B16_pretrained")
        test_v6_metrics(name_model = "faces_ViT_base_S32_v6_30-01-2024", epoch = 15, model_type="ViT_base_S32")
        test_v6_metrics(name_model = "faces_ViT_base_S16_v6_30-01-2024", epoch = 37, model_type="ViT_base_S16", patch_size=16)
        test_v6_metrics(name_model = "faces_ViT_base_S32_training+_v6_31-01-2024", epoch = 141, model_type="ViT_base_S32")
        
       
        train_test_v6_metrics(model_type="ViT_pretrained_timm")
        train_test_v6_metrics(model_type="ViT_pretrained_timm")
        
        #                                           v7
        train_v7_scenario()
        test_v7_metrics("faces_ViTEA_timm_v7_06-02-2024", 23)
        
        train_v7_scenario_separately()
        test_v7_metrics("faces_ViTEA_timm_v7_07-02-2024", 21)
        
        train_v7_scenario(add_name="DeiT_tiny", prog_model=3)
        test_v7_metrics(prog_model=3, name_model="faces_ViTEA_timm_DeiT_tiny_v7_12-02-2024", epoch=34, epoch_ae=34)
        
        train_v7_scenario(add_name="DeiT_small", prog_model=2)
        test_v7_metrics(prog_model=2, name_model="faces_ViTEA_timm_DeiT_small_v7_12-02-2024", epoch=39, epoch_ae=39)
        
        train_v7_scenario_separately(prog_vit=3, add_name="DeiT_tiny_separateTrain")
        test_v7_metrics(prog_model=3, name_folder_train="faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_13-02-2024", epoch=9)
        test_v7_metrics(prog_model=3, name_folder_train="faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_13-02-2024", epoch=25) 
        
        train_v7_scenario_separately(prog_vit=3, add_name="DeiT_tiny_separateTrain")   # longer training, same as U-net based model
        test_v7_metrics(name_folder_train= "faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_17-02-2024", epoch=22)
        test_v7_metrics(name_folder_train= "faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_17-02-2024", epoch=65)
        
        
    # GAN:
        #                                           v7
        
    # MIX:
        #                                           v7


    
    """