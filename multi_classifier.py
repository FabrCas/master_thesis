from    time                                import time
import  random              
from    tqdm                                import tqdm
from    datetime                            import date

import  torch                               as T
import  numpy                               as np
import  os              

from    torch.nn                            import functional as F
from    torch.optim                         import Adam, lr_scheduler
from    torch.cuda.amp                      import GradScaler, autocast
from    torch.utils.data                    import DataLoader
from    torchvision.transforms              import v2
from    torch.utils.data                    import default_collate

from    utilities                           import ExpLogger, loadModel, duration, check_folder, metrics_multiClass
from    dataset                             import CDDB_Partial
"""
                        Multi-class Deepfake classification models trained on CDDB dataset
"""


class MultiClassifier(object):
    """ Multi-class deepfake classifier superclass """
    
    def __init__(self, useGPU:bool, batch_size:int, model_type:str, real_grouping:str = "single"):
        super(MultiClassifier, self).__init__()
        self.model_type     = model_type
        self.useGPU         = useGPU
        self.batch_size     = batch_size
        self.real_grouping  = real_grouping
        # path 2 save
        
        self.path_models    = "./models/multi_class"
        self.path_results   = "./results/multi_class"
        
        if self.real_grouping == "single":
            # default folder name without any extension
            pass 
        elif self.real_grouping == "categories":
            self.path_models    += "_c"
            self.path_results   += "_c"
        elif self.real_grouping == "models":
            self.path_models    += "_m"
            self.path_results   += "_m"
        else:
            raise ValueError('Invalid grouping modality for real samples selected. Valid values are: "single","categories ","models"')
        
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        self.modelEpochs = 0        # variable used to store the actual number of epochs used to learn the model
        
        # initialize None variables
        self.model = None
        self.classifier_name    = None # name initially None, this changes if trained or loaded

    def compute_class_weights(self, verbose = False, multiplier = 1, normalize = True):

        print("\n\t\t[Computing Real/Fake class weights]\n")
        
        # set modality to load just the label
        self.train_dataset.set_only_labels(True)
        loader =  DataLoader(self.train_dataset, batch_size= None, num_workers= 8)  # self.dataset_train is instance of OOD_dataset class

        # compute occurrences of labels
        class_freq={}
        total = len(self.train_dataset)
        self.samples_train = total

    
        for y in tqdm(loader, total = len(loader)):
            

            try:                # encoding
                y = y.detach().cpu().tolist()
                
                # from one-hot encoding to label (positive one is realted to fake samples in second position) 
                l = y[1]
            except:             # label index
                l = y
                
            
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
        self.train_dataset.set_only_labels(False)
        
        return class_weights

    def init_logger(self, path_model, add2name = ""):
        """
            path_model -> specific path of the current model training
        """
        
        logger = ExpLogger(path_model=path_model, add2name=add2name)
        logger.write_config(self._dataConf())
        logger.write_hyper(self._hyperParams())
        try:
            logger.write_model(self.model.getSummary(verbose=False))
        except:
            print("Impossible to retrieve the model structure for logging")
        
        return logger
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def focal_loss(self, y_pred, y_true, alpha=None, gamma=2, reduction='sum'):
        """
            focal loss implementation to handle to problem of unbalanced classes
            y_pred -> logits from the model
            y_true -> ground truth label for the sample (no one-hot encoding)
            alpha -> weights for the classes
            gamma -> parameter controls the rate at which the focal term decreases with increasing predicted probability
            reduction -> choose between sum or mean to reduce over results in a batch
        """

        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
        pt = T.exp(-ce_loss)

        if alpha is not None:
            # Apply class-specific alpha weights
            alpha = alpha.to(y_pred.device)
            focal_loss = alpha[y_true] * (1 - pt) ** gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** gamma * ce_loss
            
        if reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss

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
        probs       = self.softmax(logits)
        pred        = T.argmax(probs, -1)
        fake_prob   = probs[:,1]   # positive class probability (fake probability)
        
        return pred, fake_prob, logits

    def load(self, folder_model, epoch):
        
        print("\n\t\t[Loading model]\n")
        try:
            self.classifier_name    = folder_model
            self.path2model         = os.path.join(self.path_models,  folder_model, str(epoch) + ".ckpt")
            self.path2model_results = os.path.join(self.path_results, folder_model)
            self.modelEpochs         = epoch
            loadModel(self.model, self.path2model)
            self.model.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except Exception as e:
            print(e)
            print("No model: {} found for the epoch: {} in the folder: {}".format(folder_model, epoch, self.path_models))

    @duration    
    def test(self, load_data = True):
        """
            Void function that computes the binary classification metrics

            Params:
            load_data (boolean): parameter used to specify if is required to load data or has been already done, Default is True
            
            Returns:
            None 
        """
        
        if load_data: 
            self._load_data()
        
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
        metrics_multiClass(predictions, targets, predicted_probabilities, epoch_model= str(self.modelEpochs), path_save = self.path2model_results)
    
    @duration  
    def train_and_test(self, name_train):
        """ for both train and test the model
        
        Args:
            name_train (str) should include the scenario selected and the model name (i.e. ResNet50), keep this convention {scenario}_{model_name}

        """
        self._load_data()
        
        self.train(name_train)
        self.test()



if __name__ == "__main__":
    pass