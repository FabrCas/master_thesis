import  os
from    tqdm                import tqdm
import  numpy               as np
import  torch               as T
from    torch.nn            import functional as F
from    torch.utils.data    import DataLoader
from    torch.optim         import Adam
from    torch.cuda.amp      import autocast
from    sklearn.metrics     import accuracy_score

# local modules
from    dataset             import getMNIST_dataset
from    models              import FC_classifier
from    utilities           import duration, saveModel, loadModel


""" 
        A simple classifier to test the OOD classification methods
"""

class MNISTClassifier(object):
    """ simple classifier for the MNIST dataset on handwritten digits """
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
        path_model = os.path.joint(self.path_test_models, self.name_dataset, self.name_model)
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
                
                with autocast():
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
            x = T.expand(1,-1,-1,-1)
        elif len(x.shape) <= 2 or len(x.shape) >= 5:
            raise ValueError("The input shape is not compatiple, expected a batch or a single image")
        
        x = x.to(self.device)
        with T.no_grad():
            logits      = self.model.forward(x)
            probs       = self.softmax(logits, dim=1)
            pred        = T.argmax(probs, dim=1).cpu().numpy()
        
        return pred
            
                  
if __name__ == "__main__":
    classifier = MNISTClassifier()
    classifier.train()
    