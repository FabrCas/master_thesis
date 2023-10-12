from time import time
import multiprocessing as mp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch as T

def showImage(img, name= "unknown", has_color = True):
    """ plot image using matplotlib

    Args:
        img (np.array/T.Tensor/Image): image data in RGB format, [height, width, color_channel]
    """
    
    # if torch tensor convert to numpy array
    if isinstance(img, T.Tensor):
        try:
            img = img.numpy()  # image tensor of the format [C,H,W]
        except:
            img = img.detach().cpu().numpy()
            
        # move back color channel has last dimension
        img = np.moveaxis(img,0,-1)
    
    
    plt.figure()
    
    if isinstance(img, (Image.Image, np.ndarray)): # is Pillow Image istance
        
        # if numpy array check the correct order of the dimensions
        if isinstance(img, np.ndarray):
            print(img.shape)
            if has_color and img.shape[2] != 3:
                img = np.moveaxis(img,0,-1)
            elif not(has_color) and img.shape[2] != 1:
                img = np.moveaxis(img,0,-1)
        plt.title(name)       
        plt.imshow(img)
        plt.show()
    else:
        print("img data is not valid for the printing")
        

def test_num_workers(dataloader, batch_size = 32):
    """
        simple test to choose the best number of processes to use in dataloaders
        
        Args:
        dataloader (torch.Dataloader): dataloader used to test the performance
        batch_size (int): batch dimension used during the test
    """
    n_cpu = mp.cpu_count()
    print(f"This CPU has {n_cpu} cores")
    
    for num_workers in range(1, n_cpu, 2):  
        # dataloader = DataLoader(data, batch_size= batch_size, num_workers= num_workers, shuffle= False)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(dataloader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))