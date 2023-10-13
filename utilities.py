from time import time
import os 
import json
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

def saveModel(model, name_file, path_folder= "./models"):
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
      
def saveJson(path, data):
    """ save file using JSON format

    Args:
        path (str): path to the JSON file
        data (JSON like object: dict or list): data to make persistent
    """
    with open(path, "w") as file:
        json.dump(data, file, indent= 4)
    
def loadJson(path):
    """ load file using json format

    Args:
        path (str): path to the JSON file

    Returns:
        JSON like object (dict or list): JSON data from the path
    """
    with open(path, "r") as file:
        json_data = file.read()
    data =  json.loads(json_data)
    return data

def plot_cm(cm, epoch = "_", model_name = None, path_save = None, duration_timer = 2500):
    """ sava and plot the confusion matrix

    Args:
        cm (matrix-like list): confusion matrix
        epoch (str, optional): _description_. Defaults to "_".
        model_name (_type_, optional): _description_. Defaults to None.
        path_save (_type_, optional): _description_. Defaults to None.
        duration_timer (int, optional): _description_. Defaults to 2500.
    """
    
    def close_event():
        plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # initialize timer to close plot
    if duration_timer is not None: 
        timer = fig.canvas.new_timer(interval = duration_timer) # timer object with time interval in ms
        timer.add_callback(close_event)
    
    ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x= j, y = i, s= round(cm[i, j], 3), va='center', ha='center', size='xx-large')
                
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Targets', fontsize=18)
    if model_name is not None:
        plt.title('Confusion Matrix + {}'.format(model_name), fontsize=18)
    else:
        plt.title('Confusion Matrix', fontsize=18)
    if path_save is not None:
        
        # check if the folder exists otherwise create it
        if (not os.path.exists(path_save)):
            os.makedirs(path_save)
        
        
        plt.savefig(os.path.join(path_save, 'testingCM_'+ str(epoch) +'.png'))
    if duration_timer is not None: timer.start()
    plt.show()
    
def plot_loss(loss_array, epoch="_", model_name = None, path_save = None, duration_timer = 2500):
    """ save and plot the loss by epochs

    Args:
        loss_array (list): list of avg loss for each epoch
        epoch (str, optional): _description_. Defaults to "_".
        model_name (_type_, optional): _description_. Defaults to None.
        path_save (_type_, optional): _description_. Defaults to None.
        duration_timer (int, optional): _description_. Defaults to 2500.
    """
    def close_event():
        plt.close()
    
    # check if the folder exists otherwise create it
    if (path_save is not None) and (not os.path.exists(path_save)):
        os.makedirs(path_save)
    
    # define x axis values
    x_values = list(range(1,len(loss_array)+1))
    
    color = "green"

    # Plot the array with a continuous line color
    for i in range(len(loss_array) -1):
        plt.plot([x_values[i], x_values[i + 1]], [loss_array[i], loss_array[i + 1]], color= color , linewidth=2)
        
    # text on the plot
    # if path_save is None:       
    #     plt.xlabel('steps', fontsize=18)
    # else:
    
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    if model_name is not None:
        plt.title("Learning loss plot {}".format(model_name), fontsize=18)
    else:
        plt.title('Learning loss plot', fontsize=18)
    
    # save if you define the path
    if path_save is not None:
        plt.savefig(os.path.join(path_save, 'loss_'+ str(epoch) +'.png'))
    
    fig = plt.gcf()
    
    if duration_timer is not None:
        timer = fig.canvas.new_timer(interval=duration_timer)
        timer.add_callback(close_event)
        timer.start()
    
    plt.show()