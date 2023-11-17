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
"""
                        Multi-class Deepfake classification models trained on CDDB dataset
"""


class MultiClassifier(object):
    
    def __init__(self, useGPU, batch_size, model_type):
        super(MultiClassifier, self).__init__()
        self.model_type     = model_type
        self.useGPU         = useGPU
        self.batch_size     = batch_size
        
        # path 2 save
        self.path_models    = "./models/multi_class"
        self.path_results   = "./results/multi_class"
        
        
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        self.modelEpochs = 0        # variable used to store the actual number of epochs used to learn the model
        
        # initialize None variables
        self.model = None

    def compute_class_weights(self, verbose = False):
        
        print("Computing class weights for the training set")
        # get train dataset using single mini-batch size
        loader = None # train dataloader
        
        # compute occurrences of labels
        class_freq={}
        total = len(loader)
        for y in loader:
            l = y.item()
            if l not in class_freq.keys():
                class_freq[l] = 1
            else:
                class_freq[l] = class_freq[l]+1
        if verbose: print("class_freq -> ", class_freq)
        
        # compute the weights   
        class_weights = []
        for class_ in sorted(class_freq.keys()):
            freq = class_freq[class_]
            class_weights.append(total/freq)

        print("Class_weights-> ", class_weights)
        return class_weights

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


if __name__ == "__main__":
    #                           [Start test section] 
    pass

    #                           [End test section] 
   
    """ 
            Past test/train launched: 
    """