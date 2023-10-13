from time import time
import os 
import multiprocessing as mp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch as T
from torch.utils.data import DataLoader
from tqdm import tqdm

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
            if has_color and img.shape[2] != 3:
                img = np.moveaxis(img,0,-1)
            elif not(has_color) and img.shape[2] != 1:
                img = np.moveaxis(img,0,-1)
        plt.title(name)       
        plt.imshow(img)
        plt.show()
    else:
        print("img data is not valid for the printing")
        

def test_num_workers(dataset, batch_size = 32):
    """
        simple test to choose the best number of processes to use in dataloaders
        
        Args:
        dataloader (torch.Dataloader): dataloader used to test the performance
        batch_size (int): batch dimension used during the test
    """
    n_cpu = mp.cpu_count()
    n_samples = 500
    print(f"This CPU has {n_cpu} cores")
    
    data_workers = {}
    for num_workers in range(0, n_cpu+1, 1):  
        dataloader = DataLoader(dataset, batch_size= batch_size, num_workers= num_workers, shuffle= False)
        start = time()
        for i,data in tqdm(enumerate(dataloader), total= n_samples):
            if i == n_samples: break
            pass
        end = time()
        data_workers[num_workers] = end - start
        print("Finished with:{} [s], num_workers={}".format(end - start, num_workers))
    
    data_workers = sorted(data_workers.items(), key= lambda x: x[1])
    
    print(data_workers)
    print("best choice from the test is {}".format(data_workers[0][0]))


def _saveModel(model, name_file, path_folder= "./models"):
    """ function to save weights of pytorch model as checkpoints (dict)

    Args:
        model (nn.Module): Pytorch model
        name_file (_type_): name of the checkpoint file to be saved
        path_folder (str, optional): folder used to save the models. Defaults to "./models".
    """
    
    # name_file = 'resNet3D-'+ str(name) +'.ckpt'
    path_save = os.path.join(path_folder, name_file)
    print("Saving model to: ", path_save)
    
    # create directories for models if doesn't exist
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
        
    T.save(model.state_dict(), path_save)
    
def loadModel(model, name_file, path_folder= "./models"):
    """ function to load weights of pytorch model as checkpoints (dict)

    Args:
        model (nn.Module): Pytorch model that we want to update with the new weights
        name_file (_type_): name of the checkpoint file to be saved
        path_folder (str, optional): folder used to save the models. Defaults to "./models".
    """
    # name_file = 'resNet3D-'+ str(epoch) +'.ckpt'
    path_save = os.path.join(path_folder, name_file)
    print("Loading model from: ", path_save)
    
    ckpt = T.load(path_save)
    model.load_state_dict(ckpt)