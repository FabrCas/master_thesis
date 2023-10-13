from time import time
import torch as T
from torch.utils.data import DataLoader

from utilities import *
from dataset import CDDB_binary
from models import ResNet_ImageNet, ResNet

class DFD_BinClassifier(object):
    """
        binary classifier for deepfake detection using CDDB dataset
    """
    def __init__(self, useGPU = True, batch_size = 32, model_type = "resnet_pretrained"):
        """ init classifier

        Args:
            useGPU (bool, optional): flag to use CUDA device or cpu hardware by the model. Defaults to True.
        """
        super(DFD_BinClassifier, self).__init__()
        self.useGPU = useGPU
        self.batch_size = batch_size
        self.model_type = model_type
        
        # load dataset & dataloader
        self.train_dataset = CDDB_binary(train = True)
        self.test_dataset = CDDB_binary(train = False)
        
        # laod model
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        if model_type == "resnet_pretrained":
            self.model = ResNet_ImageNet()
        else:
            self.model = ResNet()
        self.model.to(self.device)
        


    def train(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 1, shuffle= False)
        
        
        
class OOD_BinDetector(object):
    """
        Detector for OOD data
    """
    
    def __init__(self):
        pass 
        
# [test section] 
if __name__ == "__main__":
    dataset = CDDB_binary()
    # test_num_workers(dataset, batch_size  =32)   # use n_workers = 8