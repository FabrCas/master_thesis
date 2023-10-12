from time import time
import torch as T
from torch.utils.data import DataLoader

class DFD_BinClassifier(object):
    """
        binary classifier for deepfake detection using CDDB dataset
    """
    def __init__(self, useGPU = True):
        """ init classifier

        Args:
            useGPU (bool, optional): flag to use CUDA device or cpu hardware by the model. Defaults to True.
        """
        super(DFD_BinClassifier, self).__init__()
        
        # load dataset
        self.train_dataset = None
        self.test_dataset = None
        
        # laod model
        if useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        # self.model =
        # self.model.to(self.device)
        


    def train(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 1, shuffle= False)
        
        
        
class OOD_BinDetector(object):
    """
        Detector for OOD data
    """
    
    def __init__(self):
        pass 
        
# [test section]    
