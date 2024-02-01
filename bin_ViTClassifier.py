from    time                                import time
import  random              
from    tqdm                                import tqdm
from    datetime                            import date, datetime
import  math
import  torch                               as T
import  numpy                               as np
import  os              
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from    torch.nn                            import functional as F
from    torch.optim                         import Adam, lr_scheduler
from    torch.cuda.amp                      import GradScaler, autocast
from    torch.utils.data                    import DataLoader
from    torch.autograd                      import Variable
from    torchvision.transforms              import v2
from    torch.utils.data                    import default_collate

# Local imports

from    utilities                           import plot_loss, plot_valid, saveModel, metrics_binClass, loadModel, sampleValidSet, \
                                            duration, check_folder, cutmix_image, showImage, image2int, ExpLogger
from    dataset                             import getScenarioSetting, CDDB_binary, CDDB_binary_Partial
from    models                              import ViT_base, ViT_base_2, ViT_base_3, ViT_b16_ImageNet
from    bin_classifier                      import BinaryClassifier



class DFD_BinViTClassifier_v6(BinaryClassifier):
    """
        binary classifier for deepfake detection using partial CDDB dataset for the chosen scenario configuration.
        Model used: Vision-Transformer
        This class is a modification of DFD_BinClassifier_v5 class in bin_Classifier module to work with ViTs classifier,
        no confidence and no reconstruction is included here
        
        

    """
    def __init__(self, scenario, useGPU = True, patch_size = None, emb_size = None,  batch_size = 32, model_type = "ViT_base"):  # batch_size = 32 or 64
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
            Defaults is "ViT_base". 
        """
        super(DFD_BinViTClassifier_v6, self).__init__(useGPU = useGPU, batch_size = batch_size, model_type = model_type)
        self.version = 6
        self.scenario = scenario
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.augment_data_train = True
        self.use_cutmix         = False     # problem when we are learning transformer over sequence of patches
        self.n_classes          = 2
        
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
                self.model      = ViT_base_3(**kwargs,  n_layers= 6, n_heads=4)
            elif "_s" in  model_type_check:
                self.model      = ViT_base_3(**kwargs)
            elif "_m" in model_type_check:
                self.model      = ViT_base_3(**kwargs, n_layers = 12, n_heads=12)
        
                
                # self.model = ViT_base_2(n_classes= self.n_classes, n_layers= 10, n_heads=6)
            # elif "_m" in model_type.lower().strip():
            #      self.model = ViT_base_2(n_classes= self.n_classes, n_layers= 10, n_heads=10)

                
            # TODO other specs for the network here: n layers, n heads, etc
            else:
                raise ValueError("specify the dimension of the ViT_base model")
            
        elif "vit_b16_pretrained" in model_type.lower().strip():
            self.model = ViT_b16_ImageNet(n_classes= self.n_classes)
        else:
            raise ValueError("The model type is not a Unet model")
        
        self.model.to(self.device)
        self.model.eval()
        
        self._load_data()
    
        # activation function for logits
        self.sigmoid    = T.nn.Sigmoid().cuda()
        # bce defined in the training since is necessary to compute the labels weights 

        # learning hyperparameters (default)
        self.learning_coeff         = 0.4                                               # multiplier that increases the training time
        self.lr                     = 1e-4    # 1e-3 or 1e-4
        self.n_epochs               = math.floor(50 * self.learning_coeff)
        # self.n_epochs               = 15
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

        self.train_dataset  = CDDB_binary_Partial(scenario = self.scenario, train = True,  ood = False, augment= self.augment_data_train, label_vector= True)
        
        # get valid and test sets
        test_dataset        = CDDB_binary_Partial(scenario = self.scenario, train = False, ood = False, augment= False)
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
            plot_loss(loss_epochs, title_plot= name_train, path_save = None)
            plot_valid(valid_history, title_plot= name_train, path_save = None)
        else: 
            plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
            plot_loss(loss_epochs, title_plot= name_train, path_save = os.path.join(path_model_folder,name_loss_file), show=False)
            plot_valid(valid_history, title_plot= name_train, path_save = os.path.join(path_results_folder, name_valid_file))
            plot_valid(valid_history, title_plot= name_train, path_save = os.path.join(path_model_folder,name_valid_file), show=False)
        
        # save model
        saveModel(self.model, path_model_save)
    
        # terminate the log session
        logger.end_log()

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


if __name__ == "__main__":
    #                           [Start test section] 
    
    scenario_prog       = 0
    data_scenario       = None
    
    if scenario_prog == 0: 
        data_scenario = "content"
    elif scenario_prog == 1:
        data_scenario = "group"
    
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
    
    train_test_v6_metrics(model_type="ViT_base_M16", patch_size=16)
    
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
    # GAN:
        #                                           v6


    
    """