from time import time
import multiprocessing as mp
import torch as T
from torch.utils.data import DataLoader

class DFD_binary_classifier(object):
    """
        binary classifier for deepfake detection using CDDB dataset
    """
    def __init__(self, useGPU = True):
        """ init classifier

        Args:
            useGPU (bool, optional): flag to use CUDA device or cpu hardware by the model. Defaults to True.
        """
        super(DFD_binary_classifier, self).__init__()
        
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
        
        
        
        
# [test section]    
def test_num_workers(batch_size = 32):
    """
        simple test to choose the best number of processes to use in dataloaders
    """
    data  = None 
    
    for num_workers in range(1, mp.cpu_count(), 2):  
        dataloader = DataLoader(data, batch_size= batch_size, num_workers= num_workers, shuffle= False)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(dataloader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))