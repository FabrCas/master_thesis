import  os 
import  re
import  json
import  random
import  math
from    time                                import time
import  multiprocessing                     as mp
import  numpy                               as np
from    PIL                                 import Image
import  matplotlib.pyplot                   as plt
from    datetime                            import date
import  torch                               as T
from    torch.utils.data                    import DataLoader, random_split, ConcatDataset
from    tqdm                                import tqdm
from    torchvision                         import transforms
from    torchvision.transforms              import v2    # new version for tranformation methods
from    torchvision.transforms.functional   import InterpolationMode
from    time                                import time
from    skimage.filters                     import gaussian

#  classification metrics
from    sklearn.metrics     import  precision_score, recall_score, f1_score, confusion_matrix, hamming_loss,\
                                    jaccard_score, accuracy_score, classification_report, matthews_corrcoef
#  binary classification metrics
from    sklearn.metrics     import  auc, roc_curve, average_precision_score, precision_recall_curve


################################################## Settings ###########################################################################

def get_inputConfig(spatial_dim = 112):
    # choose between images of spatial dimensions 224p of 112p, always squared resolution
    return {
        "width"     : spatial_dim,
        "height"    : spatial_dim,
        "channels"  : 3
    }


##################################################  Dataset utilties ##################################################################


def mergeDatasets(dataset1, dataset2):
    """ alias for ConcatDataset from pytorch"""
    return ConcatDataset([dataset1, dataset2])

def sampleValidSet(trainset, testset, useOnlyTest = True, verbose = False):
    """
        Function used to partiton data to have also a validatio set for training.
        The validation set is composed by the 10% of the overall amount of samples.
        you can choose to sample from both test and training set (from both is taken the 5%)
        or only from the test set (in this case is collected the 10% from only this set).
        
        
        Args:
            trainset (T.Dataset): The trainig set
            testset  (T.Dataset): The test set
            useOnlyTest (boolean): flag to select which source is used to sample the data
            verbose (boolean):    flag to active descriptive prints
            
        Returns:
            if useOnlyTest is True:
                validset  (pytorch.Dataset),
                testset   (pytorch.Dataset)
            if useOnlyTest is False:
                trainset  (pytorch.Dataset),
                validset  (pytorch.Dataset),
                testset   (pytorch.Dataset)
    """
    print("Separating data for the validation Set...")
    generator = T.Generator().manual_seed(22)

    all_data = len(trainset) + len(testset)
    
    if verbose:
        print("\ntrain length ->", len(trainset))
        print("test length ->", len(testset))
        print("total samples ->°", all_data)
        print("Data percentage distribution over sets before partition:")
        print("TrainSet [%]",round(100*(len(trainset)/all_data),2) )
        print("TestSet  [%]",round(100*(len(testset)/all_data),2),"\n")
    
    
    # if not(useTestSet) and (not trainset is None): 
    if not(useOnlyTest):
        """
            split data with the following strategy, validation set is the 10% of all data.
            These samples are extract half from training set and half from test set.
            after this we have almost the following distribution:
            training        -> 65%
            validation      -> 10%
            testing         -> 25% 
        """
        
        
        # compute relative percentage taking 5% for each set
        perc_train = round(((0.1 * all_data)*0.5/len(trainset)),3)
        perc_test  = round(((0.1 * all_data)*0.5/len(testset)),3)

        
        
        trainset, val_p1  = random_split(trainset,  [1-perc_train, perc_train], generator = generator)  #[0.92, 0.08]
        if verbose: print(f"splitting train (- {perc_train}%) ->",len(val_p1), len(trainset))

        testset,  val_p2  = random_split(testset,  [1-perc_test, perc_test], generator = generator)   #[0.84, 0.16]
        if verbose: print(f"splitting test (- {perc_test}%) ->",len(val_p2), len(testset))
        
        validset = ConcatDataset([val_p1, val_p2])
        if verbose: print("validation length ->", len(validset))
        
        if verbose:
            print("\nData percentage distribution over sets after partition:")
            print("TrainSet [%]",round(100*(len(trainset)/all_data),2) )
            print("TestSet  [%]",round(100*(len(testset)/all_data),2)  )
            print("ValidSet [%]",round(100*(len(validset)/all_data),2)   )
        
        return trainset, validset, testset
    else:
        """
            split data with the following strategy, validation set is the 10% of all data.
            These samples are extract all from the test set.
        """
        
        perc_test = round(((0.1 * all_data)/len(testset)),3)
        testset,  validset  = random_split(testset,  [1-perc_test, perc_test], generator = generator)
        
        if verbose:
            print(f"splitting test (- {perc_test}%) ->",len(validset), len(testset))
            print("validation length", len(validset))

        if verbose:
            print("\nData percentage distribution over sets after partition:")
            print("TrainSet [%]",round(100*(len(trainset)/all_data),2) )
            print("TestSet  [%]",round(100*(len(testset)/all_data),2)  )
            print("ValidSet [%]",round(100*(len(validset)/all_data),2)   )

        return validset, testset
        
def balanceLabels(self, dataloader, verbose = False):
    """ 
        used to make the dataset more balanced (downsampling)
    """
    
    # compute occurrences of labels
    class_freq={}
    labels = []
    
    total = len(dataloader)
    for y  in tqdm(dataloader, total= total):
        l = y.item()
        if l not in class_freq.keys():
            class_freq[l] = 1
        else:
            class_freq[l] = class_freq[l]+1
        labels.append(l)
    
    n_labels = len(labels)   
    
    # sorting the dictionary by key
    class_freq = {k: class_freq[k] for k in sorted(class_freq.keys())}
    
    if verbose: print("class frequency: ->", class_freq)
    
    min_freq = min(class_freq.items(), key= lambda x: x[1])
    if verbose: print("minimum frequency ->", min_freq)
    
    max_freq = max(class_freq.items(), key= lambda x: x[1])
    if verbose: print("maximum frequency ->", max_freq)
    
    indices_0 = [idx for idx, val in enumerate(labels) if val == 0]
    indices_1 = [idx for idx, val in enumerate(labels) if val == 1]
    indices_2 = [idx for idx, val in enumerate(labels) if val == 2]
    indices_3 = [idx for idx, val in enumerate(labels) if val == 3]
    
    # complete list of indices
    indices = {0:indices_0, 1:indices_1, 2:indices_2, 3:indices_3}
    
    if verbose: print("lenght indices:", len(indices_0), len(indices_1), len(indices_2), len(indices_3))

    # dict of sampled indices
    sampled_indices = {}
    
    for k,v in class_freq.items():
        if k != min_freq[0]:
            multiplier = ((n_labels - v)/(n_labels)) * 0.25
            
            
            # 20% of the actual values + the number of minimum samples
            n_sample = int(v*multiplier) # + min_freq[1]
            if n_sample > v:
                n_sample = v
            
            if verbose: print(f"n_sample for {k} -> {n_sample}")
            sampled_indices[k] = random.sample(indices[k], n_sample)
        else:
            if verbose: print(f"n_sample for {k} -> {v}")
            
            sampled_indices[k] = indices[k]
        

    # build the flat list with the sorted indices selected
    final_list = []
    for k,v in sampled_indices.items():
        final_list = [*final_list, *v] 
    
    final_list = sorted(final_list)
    
    if verbose: print("number of samples after the reduction: ", len(final_list))

    return class_freq, final_list

##################################################  image transformation/data augmentation ############################################

def get_transformationInput(isTensor = False):
    """ function that returns trasnformation operations sequence for the image input to be compatible for ResNet50 model

    Returns:
        compose pytorch object
    """
    
    config = get_inputConfig()
    w = config['width']
    h = config['height']
    
    
    if isTensor:
        transform_ops = transforms.Resize((h, w), interpolation= InterpolationMode.BILINEAR, antialias= True),
    else: 
        transform_ops = transforms.Compose([
            transforms.Resize((h, w), interpolation= InterpolationMode.BILINEAR, antialias= True),
            transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
            lambda x: T.clamp(x, 0, 1),
            # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])   # normlization between -1 and 1, using the whole range uniformly, formula: (pixel - mean)/std
        ])
    
    return transform_ops

def augment_v1(x):
    
    config = get_inputConfig()
    w = config['width']
    h = config['height']
    
    x = v2.ToTensor()(x)   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
    x = v2.Resize((w, h), interpolation= InterpolationMode.BILINEAR, antialias= True)(x)
    # x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
    x = v2.RandomHorizontalFlip(0.5)(x)
    x = v2.RandomVerticalFlip(0.1)(x)
    x = v2.RandAugment(num_ops = 1, magnitude= 7, num_magnitude_bins= 51, interpolation = InterpolationMode.BILINEAR)(x)
    x = T.clamp(x, 0, 1)
    
    return x

def image2int(img_tensor, is_range_zero_one = True):
    """ 
        convert img (T.tensor) to int range 0-255. img can be a single image or a batch
        the img is float, if range is [0,1] set is_range_zero_one = True
        if range is [-1,1] set is_range_zero_one = False 
    """
    
    # auto_check for range
    if T.min(img_tensor) < 0:               # sufficient but not necessary condition (if min > 0 could be in both ranges)
        is_range_zero_one = False    
    
    if len(img_tensor.shape) == 4 or len(img_tensor.shape) == 3:
        if not(is_range_zero_one):
            img_tensor = (img_tensor+1)/2 # convert range [-1, 1] to [0,1]
        img_tensor = img_tensor * 255
        # img_tensor = img_tensor.to(T.int32)
        return img_tensor
    else:
        raise ValueError(f"Unsupported img_tensor shape, is: {img_tensor.shape}")
    
# cutmix technique: https://arxiv.org/abs/1905.04899

def rand_bbox(size, lamb):
    """ Generate random bounding box 
    Args:
        - size: [width, breadth] of the bounding box
        - lamb: (lambda) cut ratio parameter, sampled from Beta distribution
    Returns:
        - Bounding box
    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lamb)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_image(image_batch, image_batch_labels, beta = 1):
    """ Generate a CutMix augmented image from a batch 
    Args:
        - image_batch (numpy.Array): a batch of input images
        - image_batch_labels (numpy.Array): labels corresponding to the image batch
        - beta (int): a parameter of Beta distribution.
    Returns:
        - CutMix image batch, updated labels
    """
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = np.random.permutation(len(image_batch))
    target_a = image_batch_labels
    target_b = image_batch_labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox((image_batch.shape[2], image_batch.shape[3]), lam)
    # print(bbx1, bby1, bbx2, bby2)
    image_batch_updated = image_batch.copy()
    
    # image_batch_updated[:, bbx1:bbx2, bby1:bby2, :] = image_batch[rand_index, bbx1:bbx2, bby1:bby2, :]
    image_batch_updated[:, :, bbx1:bbx2, bby1:bby2] = image_batch[rand_index, :, bbx1:bbx2, bby1:bby2]
    # image_batch_updated[:, :, 100:150, 100:150] = image_batch[rand_index, :, 100:150, 100:150]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_batch.shape[2] * image_batch.shape[3]))
    label = target_a * lam + target_b * (1. - lam)
    
    return image_batch_updated, label

# data normalization

class NormalizeByChannelMeanStd(T.nn.Module):
    def __init__(self, data):   # data i.e trainig set 
        super(NormalizeByChannelMeanStd, self).__init__()
        # if not isinstance(mean, T.Tensor):
        #     mean = T.tensor(mean)
        # if not isinstance(std, T.Tensor):
        #     std = T.tensor(std)
        
        if not isinstance(data, T.Tensor):
            data = T.tensor(data)

        self._compute_mean_std(data)
        self.register_buffer("mean", self.mean)
        self.register_buffer("std", self.std)

    def _compute_mean_std(self, data): 
        self.mean   = T.mean(data)
        self.std    = T.std(data)
    
    def forward(self, tensor):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1, so: [batch_size, color_channel, height, width]
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(self.mean).div(self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

# data perturbation

def normalize(x):
    # return (x - np.min(x)) / (np.max(x) - np.min(x))
    return x 

def add_noise(x, complexity=0.5):
    mean = 0
    std = 1e-9 + complexity
    # return batch_input +np.random.normal(size=batch_input.shape, scale=1e-9 + complexity)
    return x + T.rand_like(x) * std + mean

def add_blur(img, complexity=0.5):
    """ img as fist dimension we have the color channel."""
    # image = img.reshape((-1, 28, 28))
    return gaussian(img, sigma=5*complexity, channel_axis=-1)

def add_distortion_noise(x):
    distortion = np.random.uniform(low=0.9, high=1.2)
    mean = 0
    std = 1e-9 + distortion
    return x + T.rand_like(x) * std + mean
    
    # return batch_input + np.random.normal(size=batch_input.shape, scale=1e-9 + distortion)
    
##################################################  Save/Load functions ###############################################################

def saveModel(model, path_save):
    """ function to save weights of pytorch model as checkpoints (dict)

    Args:
        model (nn.Module): Pytorch model
        path_save (str): path of the checkpoint file to be saved
    """
    print("Saving model to: ", path_save)        
    T.save(model.state_dict(), path_save)
    
def loadModel(model, path_load):
    """ function to load weights of pytorch model as checkpoints (dict)

    Args:
        model (nn.Module): Pytorch model that we want to update with the new weights
        path_load (str): path to locate the model weights file (.ckpt)
    """
    print("Loading model from: ", path_load)
    ckpt = T.load(path_load)
    model.load_state_dict(ckpt)
      
def saveJson(path_file, data):
    """ save file using JSON format

    Args:
        path_file (str): path to the JSON file
        data (JSON like object: dict or list): data to make persistent
    """
    print(f"Saving json file: {path_file}")
    with open(path_file, "w") as file:
        json.dump(data, file, indent= 4)
    
def loadJson(path_file):
    """ load file using json format

    Args:
        path (str): path to the JSON file

    Returns:
        JSON like object (dict or list): JSON data from the path
    """
    
    print(f"Loading json file: {path_file}")
    with open(path_file, "r") as file:
        json_data = file.read()
    data =  json.loads(json_data)
    return data

##################################################  Plot/show functions ###############################################################

def showImage(img, name= "unknown", has_color = True, save_image = False):
    """ plot image using matplotlib

    Args:
        img (np.array/T.Tensor/Image): image data in RGB format, [height, width, color_channel]
    """
    
    if save_image:
        check_folder("./static/saved_imgs")
        if name == "unknown":
            name += "_" + date.today().strftime("%d-%m-%Y") + ".png"
    

    # if torch tensor convert to numpy array
    if isinstance(img, T.Tensor):
        try:
            img = img.numpy()  # image tensor of the format [C,H,W]
        except:
            img = img.detach().cpu().numpy()
            
        # move back color channel has last dimension, (in Tensor the convention for the color channel is to use the first after the batch)
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
        if has_color:
            plt.imshow(img)
        else: 
            plt.imshow(img, cmap="gray")
        if save_image: plt.savefig(os.path.join("./static/saved_imgs", name))
        plt.show()
    else:
        print("img data is not valid for the printing")

def show_img_grid(imgs, titles):  # in alternative look at torchvision.utils.make_grid(images)
    
    def show_img(img, ax=None, title=None):
        """Shows a single image."""
        if ax is None:
            ax = plt.gca()
        ax.imshow(img[...])
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)
            
    """ Shows a grid of images. imgs a list of numpy array"""
    n = int(np.ceil(len(imgs)**.5))
    _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        # img = (img + 1) / 2  # Denormalize if -1< 1 is the interval
        show_img(img, axs[i // n][i % n], title)

def plot_loss(loss_array, title_plot = None, path_save = None, duration_timer = 2500, show = True):
    """ save and plot the loss by epochs

    Args:
        loss_array (list): list of avg loss for each epoch
        title_plot (str, optional): _title to exhibit on the plot
        path_save (str, optional): relative path for the location of the save folder
        duration_timer (int, optional): milliseconds used to show the plot 
    """
    def close_event():
        plt.close()
    
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
    if title_plot is not None:
        plt.title("Learning loss plot {}".format(title_plot), fontsize=18)
    else:
        plt.title('Learning loss plot', fontsize=18)
    
    # save if you define the path
    if path_save is not None:
        plt.savefig(path_save)
    
    fig = plt.gcf()
    
    if duration_timer is not None:
        timer = fig.canvas.new_timer(interval=duration_timer)
        timer.add_callback(close_event)
        timer.start()
    if show:
        plt.show(block = True)
    
    plt.clf()  # clear plot data
        
def plot_valid(valid_history, title_plot = None, path_save = None, duration_timer = 2500, show = True):
    """ save and plot the loss by epochs

    Args:
        valid_history (list): list of valid criterion by epochs 
        title_plot (str, optional): _title to exhibit on the plot
        path_save (str, optional): relative path for the location of the save folder
        duration_timer (int, optional): milliseconds used to show the plot 
    """
    def close_event():
        plt.close()
    
    # define x axis values
    x_values = list(range(1,len(valid_history)+1))
    
    color = "blue"

    # Plot the array with a continuous line color
    for i in range(len(valid_history) -1):
        plt.plot([x_values[i], x_values[i + 1]], [valid_history[i], valid_history[i + 1]], color= color , linewidth=2)
        
    # text on the plot
    # if path_save is None:       
    #     plt.xlabel('steps', fontsize=18)
    # else:
    
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    if title_plot is not None:
        plt.title("Valid history {}".format(title_plot), fontsize=18)
    else:
        plt.title('Valid history', fontsize=18)
    
    # save if you define the path
    if path_save is not None:
        plt.savefig(path_save)
    
    fig = plt.gcf()
    
    if duration_timer is not None:
        timer = fig.canvas.new_timer(interval=duration_timer)
        timer.add_callback(close_event)
        timer.start()
    if show:
        plt.show(block = True)
        
    plt.clf()  # clear plot data

def plot_cm(cm, labels, title_plot = None, path_save = None, duration_timer = 2500):
    """ sava and plot the confusion matrix

    Args:
        cm (matrix-like list): confusion matrix
        labels (list) : labels to index the matrix
        title_plot (str, optional): text to visaulize as title of the plot
        path_save (str, optional): relative path for the location of the save folder
        duration_timer (int, optional): milliseconds used to show the plot 
    """
    
    def close_event():
        plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x= j, y = i, s= round(cm[i, j], 3), va='center', ha='center', size='xx-large')
        
    # change labels name on the matrix
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize = 9)
    ax.set_yticklabels(labels, fontsize = 9)
            
    plt.xlabel('Predictions', fontsize=11)
    plt.ylabel('Targets', fontsize=11)
    
    if title_plot is not None:
        plt.title('Confusion Matrix + {}'.format(title_plot), fontsize=18)
    else:
        plt.title('Confusion Matrix', fontsize=18)
        
    if path_save is not None:
        plt.savefig(os.path.join(path_save, "confusion_matrix.png"))
    

    # initialize timer to close plot
    if duration_timer is not None: 
        timer = fig.canvas.new_timer(interval = duration_timer) # timer object with time interval in ms
        timer.add_callback(close_event)
        timer.start()
        
    plt.show()
        
    # if duration_timer is not None: timer.start()
    # else:
    #    
    
def plot_ROC_curve(fpr, tpr, path_save = None, duration_timer = 2500):
    def close_event():
        plt.close()
        # plt.clf()  # clear plot data
    
    plt.figure()
    plt.plot(fpr, tpr, color='lime', lw=2, label=f'ROC curve)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    
    if path_save is not None:
        plt.savefig(os.path.join(path_save, "ROC_curve.png"))
        
    fig = plt.gcf()
    
    if duration_timer is not None:
        timer = fig.canvas.new_timer(interval=duration_timer)
        timer.add_callback(close_event)
        timer.start()

    plt.show()
    # plt.clf()  # clear plot data
    
def plot_PR_curve(recalls, precisions, path_save = None, duration_timer = 2500):
   
    def close_event():
        plt.close()
        # plt.clf()  # clear plot data
    
    plt.figure()
    plt.plot(recalls, precisions, color='mediumblue', lw=2, label=f'PR curve)')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall curve')
    plt.legend(loc='lower right')
    
    if path_save is not None:
        plt.savefig(os.path.join(path_save, "PR_curve.png"))
        
    fig = plt.gcf()
    
    if duration_timer is not None:
        timer = fig.canvas.new_timer(interval=duration_timer)
        timer.add_callback(close_event)
        timer.start()
    # else:
    #     plt.clf()  # clear plot data
    plt.show()
        
    

##################################################  Metrics functions #################################################################

def metrics_binClass(preds, targets, pred_probs, epoch_model="unknown", path_save = None, average = "macro",  save = True, name_ood_file  = None, ood_labels_reversed = False):
    """ 
        Computation of metrics for binary classification tasks.
        use name_ood_file parameter to save for OOD binary classification, otherwise is saved for classic binary classification
    """
    
    if pred_probs is not None:
        # roc curve/auroc computation
        fpr, tpr, thresholds = roc_curve(targets, pred_probs, pos_label=1,  drop_intermediate= False)
        roc_auc = auc(fpr, tpr)
        
        # plot and save ROC curve
        plot_ROC_curve(fpr, tpr, path_save)
        
        # pr curve/ aupr computation
        p,r, thresholds = precision_recall_curve(targets, pred_probs, pos_label=1,  drop_intermediate= False)
        aupr = auc(r,p)    # almost the same of average precision score
        
        # plot PR curve
        plot_PR_curve(r,p,path_save) 
        
        # compute everage precision (almost equal to AUPR)
        avg_precision = average_precision_score(targets, pred_probs, average= average, pos_label=1),         \
    
    else: 
        avg_precision  = "empty"
        roc_auc        = "empty"
        aupr           = "empty"
    
    # compute metrics and store into a dictionary
    metrics_results = {
        "accuracy":                     accuracy_score(targets, preds, normalize= True),
        "precision":                    precision_score(targets, preds, average = "binary", zero_division=1, pos_label=1),   \
        "recall":                       recall_score(targets, preds, average = "binary", zero_division=1, pos_label=1),      \
        "f1-score":                     f1_score(targets, preds, average= "binary", zero_division=1, pos_label=1),           \
        "average_precision":           avg_precision,                                                                        \
        "ROC_AUC":                      roc_auc,                                                                             \
        "PR_AUC":                       aupr,                                                                                \
        "hamming_loss":                 hamming_loss(targets, preds),                                                        \
        "jaccard_score":                jaccard_score(targets,preds, pos_label=1, average="binary", zero_division=1),        \
        "confusion_matrix":             confusion_matrix(targets, preds, labels=[0,1], normalize="true")
         
    }
    
    # print metrics
    for k,v in metrics_results.items():
        if k != "confusion_matrix":
            print("\nmetric: {}, result: {}".format(k,v))
        else:
            print("\nConfusion matrix:")
            print(v)
    
    
    # plot and save (if path specified) confusion matrix
    if name_ood_file is None:
        plot_cm(cm = metrics_results['confusion_matrix'], labels = ["real", "fake"], title_plot = None, path_save = path_save)
    else:
        if ood_labels_reversed:
            plot_cm(cm = metrics_results['confusion_matrix'], labels = ["ID", "OOD"], title_plot = None, path_save = path_save)
        else:
            plot_cm(cm = metrics_results['confusion_matrix'], labels = ["OOD", "ID"], title_plot = None, path_save = path_save)
    
    print(path_save)
    
    metrics_results['confusion_matrix'] = metrics_results['confusion_matrix'].tolist()
    
    
    # save the results (JSON file) if a path has been provided
    if path_save is not None and save:
        # metrics_results_ = metrics_results.copy()
        # metrics_results_['confusion_matrix'] = metrics_results_['confusion_matrix'].tolist()
        
        if name_ood_file is None:    # save for binary classification real/fake
            saveJson(os.path.join(path_save, 'binaryMetrics_' + epoch_model + '.json'), metrics_results)
        else:                       # save for binary classification ID/OOD
            saveJson(os.path.join(path_save, name_ood_file), metrics_results)
    
    return metrics_results

def metrics_OOD(targets, pred_probs, pos_label = 1, path_save_json = None, path_save_plot = None, save = True):
    """ 
        Computation of metrics for the OOD classification task
    """
    fpr, tpr, _ = roc_curve(targets, pred_probs, pos_label=1,  drop_intermediate= False)
    auroc = auc(fpr, tpr)
    
    # plot and save ROC curve
    plot_ROC_curve(fpr, tpr, path_save_plot, duration_timer = None)
    
    fpr95 = fpr_at_95_tpr(pred_probs, targets, pos_label)
    
    det_err, thr_err = detection_error(pred_probs, targets, pos_label= pos_label)
    
    metric_results = {
        "auroc":            auroc,
        "fpr95":            fpr95,
        "detection_error":  det_err,
        "thr_de":           thr_err
    }
    
    # save the results (JSON file) if a path has been provided
    if path_save_json is not None and save:
        saveJson(os.path.join(path_save_json, 'metricsOOD.json'), metric_results)
    
    return metric_results

def fpr_at_95_tpr(preds, labels, pos_label = 1):
    '''
    Returns the false positive rate when the true positive rate is at minimum 95%.
    '''
    fpr, tpr, _ = roc_curve(labels, preds, pos_label= pos_label, drop_intermediate= True)
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(x = 0.95, xp = tpr,fp = fpr)   # x->y(x) interpolated, xp->[x1,..,xn], fp-> [y1,..-,yn]

def detection_error(preds, labels, pos_label = 1, verbose = False):   # look also det_curve from sklearn metrics
    '''
    Return the misclassification probability when TPR is 95%.
    '''
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label= pos_label, drop_intermediate= True)
    
    if verbose:
        for i in range(len(fpr)):
            print(f"thr:{thresholds[i]}, fpr:{fpr[i]}, tpr:{tpr[i]}")
    
    # Get ratio of true positives to false positives
    pos_ratio = sum(np.array(labels) == pos_label) / len(labels)
    neg_ratio = 1 - pos_ratio

    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x >= 0.95]

    # Calc error for a given threshold (i.e. idx)
    _detection_error = lambda idx: neg_ratio * (1 - tpr[idx]) + pos_ratio * fpr[idx]

    # Find the minimum detection error such such that TPR >= 0.95
    min_error_idx = min(idxs, key=_detection_error)
    
    detection_error = _detection_error(min_error_idx)
    
    threshold_de = thresholds[min_error_idx]
    
    
    # Return the minimum detection error such that TPR >= 0.95
    # detection_error, index = min(map(_detection_error, idxs))
    # index = idxs
    
    return detection_error, threshold_de

def metrics_multiClass(preds, targets, labels_indices, labels_name, average = "macro", epoch_model="unknown", path_save = None, save = True):
    """ 
        Computation of metrics for multi-label classification task.
        "average" parameter, choose between "macro" (over classes) and "micro" (over samples)
        labels_indices and labels_name should be in the same order of representation
    """
    
    # compute classification report
    report = classification_report(y_true=targets, y_pred=preds, labels = labels_indices, target_names= labels_name, output_dict= True)
    
    # compute metrics and store into a dictionary
    metrics_results = {
        "accuracy"          : accuracy_score(targets, preds, normalize= True),
        "precision"         : precision_score(targets, preds, average = average, zero_division=1),      \
        "recall"            : recall_score(targets, preds, average = average, zero_division=1),         \
        "f1-score"          : f1_score(targets, preds, average= average, zero_division=1),              \
        "hamming_loss"      : hamming_loss(targets, preds),                                             \
        "jaccard_score"     : jaccard_score(targets,preds, average = average, zero_division=1),         \
        "matthew_correlation_coefficient"                                                              
                            : matthews_corrcoef(targets, preds),                                        \
        "confusion_matrix"  : confusion_matrix(targets, preds, labels=labels_indices, normalize="true") \
         
    }
    
    # concat metrics computed
    metrics_results = concat_dict(metrics_results, report)
    
    # print metrics
    for k,v in metrics_results.items():
        if k != "confusion_matrix":
            print("\nmetric: {}, result: {}".format(k,v))
        else:
            print("Confusion matrix")
            print(v)
    
    
    # plot and save (if path specified) confusion matrix
    plot_cm(cm = metrics_results['confusion_matrix'], labels = labels_name, title_plot = None, path_save = path_save)
    
    
    metrics_results['confusion_matrix'] = metrics_results['confusion_matrix'].tolist()
    
    # save the results (JSON file) if a path has been provided
    if path_save is not None and save:
        # metrics_results_ = metrics_results.copy()
        # metrics_results_['confusion_matrix'] = metrics_results_['confusion_matrix'].tolist()
        saveJson(os.path.join(path_save, 'binaryMetrics_' + epoch_model + '.json'), metrics_results)

    
    return metrics_results

##################################################  performance testing functions #####################################################

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

# use as function decorator
def duration(function):
    def wrapper_duration(*args, **kwargs):
        t_start = time()
        function(*args, **kwargs)   # to unpack no-keywords (*args) and keywords (**) arguments
        print("Total time elapsed for the execution of {} function: {} [s]".format(function.__name__, round((time() - t_start),4))) 
    return wrapper_duration

    """  how to use
    @duration
    def example(start,stop,power):
        a = list(range(start,stop))
        pow = lambda x: x**power
        result = list(map(pow, a))
        print(result)
    """

##################################################  Neural Network utilities ##########################################################

#                                       get dimensions and n°parameters for layers
# 2D Convolutional layer
def conv2d_shapes(input_shape, n_filters, kernel_size, padding = 0, stride = 1):
    """ get info on output and n°learnable params from a convolutional 2D layer

    Args:
        input_shape (tuple): shape dimension of the input
        n_filters (int): number of filter used
        kernel_size (int): size kernel for width and height filter
        padding (int, optional): padding for the input to include in the input shape. Default is 0
        stride (int, optional): stride to use in the convolution.  Default is 1
    """
    conv2d_out_shape(input_shape, n_filters, kernel_size, padding = padding, stride = stride)
    conv2d_n_parameters(input_shape, n_filters, kernel_size)

def conv2d_out_shape(input_shape, n_filters, kernel_size, padding, stride):
    # input_shape: [batch, channels, height, width] or [channels, height, width], is a tuple
    print("Dimensions after 2D Convolutional layer application, with filter size: {}x{}, stride: {} and padding: {}\n".format(kernel_size, kernel_size, stride, padding))
    c_out = n_filters
    if len(input_shape) > 3:  
        # batch case
        print("The input shape is ->\t\t[batch: {}, channels: {}, height: {}, width: {}]".format(input_shape[0], str(input_shape[1]), str(input_shape[2]), str(input_shape[3])))
        print("The filter (x{}) shape is ->\t[depth: {}, height: {}, width: {}]:".format(str(n_filters), str(input_shape[1]), str(kernel_size), str(kernel_size)))
        h_out = int((math.floor(input_shape[2] - kernel_size + 2*padding )/stride) + 1)
        w_out = int((math.floor(input_shape[3] - kernel_size + 2*padding )/stride) + 1)
        print("The output shape is ->\t\t[batch: {}, channels: {}, height: {}, width: {}]".format(str(input_shape[0]), str(c_out), str(h_out), str(w_out)))
        return (input_shape[0], c_out, h_out, w_out)
        
    else:
        # single image case              
        print("The input shape is -> \t\t[channels: {}, height: {}, width: {}]".format(str(input_shape[0]), str(input_shape[1]), str(input_shape[2])))
        print("The filter (x{}) shape is ->\t[depth: {}, height: {}, width: {}]:".format(str(n_filters), str(input_shape[0]), str(kernel_size), str(kernel_size)))
        h_out = int((math.floor(input_shape[1] - kernel_size + 2*padding )/stride) + 1)
        w_out = int((math.floor(input_shape[2] - kernel_size + 2*padding )/stride) + 1)
        print("The output shape is ->\t\t[channels: {}, height: {}, width: {}]".format(str(c_out), str(h_out), str(w_out)))
        return (c_out, h_out, w_out)
        
def conv2d_n_parameters(input_shape, n_filters, kernel_size):
    if len(input_shape) > 3: 
        depth_filter = input_shape[1]
    else:
        depth_filter = input_shape[0]
    
    learnable_params = (kernel_size**2 * depth_filter +1) * n_filters
    print("The number of learnable parameters is -> {}".format(learnable_params))
    return learnable_params 

# 2D Convolutioal transposed layer (for more on conv2d layer look at: https://medium.com/analytics-vidhya/demystify-transposed-convolutional-layers-6f7b61485454)
def convTranspose2d_shapes(input_shape, n_filters, kernel_size, padding = 0, stride = 1, output_padding = 0):
    """ get info on output and n°learnable params from a convolutional transpose 2D layer

    Args:
        input_shape (tuple): shape dimension of the input
        n_filters (int): number of filter used
        kernel_size (int): size kernel for width and height filter
        padding (int, optional): padding for the input to include in the input shape
        output_padding (int, optional): padding for the input to include in the input shape. Default is 0
        stride (int, optional): stride to use in the convolution. Default is 1
    """
    
    convTranspose2d_out_shapes(input_shape, n_filters, kernel_size, padding = padding, stride = stride, output_padding = output_padding)
    convTranspose2d_n_parameters(input_shape, n_filters, kernel_size)
    
def convTranspose2d_out_shapes(input_shape, n_filters, kernel_size, padding, stride, output_padding):
    print("Dimensions after 2D Convolutional Transpose layer application, with filter size: {}x{}, stride: {} and padding: {}\n".format(kernel_size, kernel_size, stride, padding))
    c_out = n_filters
    if len(input_shape) > 3:  
        # batch case
        print("The input shape is ->\t\t[batch: {}, channels: {}, height: {}, width: {}]".format(input_shape[0], str(input_shape[1]), str(input_shape[2]), str(input_shape[3])))
        print("The filter (x{}) shape is ->\t[depth: {}, height: {}, width: {}]:".format(str(n_filters), str(input_shape[1]), str(kernel_size), str(kernel_size)))
        h_out = int(math.floor( (input_shape[2] - 1)*stride + kernel_size - 2*padding + output_padding ))
        w_out = int(math.floor( (input_shape[3] - 1)*stride + kernel_size - 2*padding + output_padding ))
        print("The output shape is ->\t\t[batch: {}, channels: {}, height: {}, width: {}]".format(str(input_shape[0]), str(c_out), str(h_out), str(w_out)))
        return (input_shape[0], c_out, h_out, w_out)
        
    else:
        # single image case              
        print("The input shape is -> \t\t[channels: {}, height: {}, width: {}]".format(str(input_shape[0]), str(input_shape[1]), str(input_shape[2])))
        print("The filter (x{}) shape is ->\t[depth: {}, height: {}, width: {}]:".format(str(n_filters), str(input_shape[0]), str(kernel_size), str(kernel_size)))
        h_out = int(math.floor( (input_shape[1] - 1)*stride + kernel_size - 2*padding + output_padding ))
        w_out = int(math.floor( (input_shape[2] - 1)*stride + kernel_size - 2*padding + output_padding ))
        print("The output shape is ->\t\t[channels: {}, height: {}, width: {}]".format(str(c_out), str(h_out), str(w_out)))
        return (c_out, h_out, w_out)

def convTranspose2d_n_parameters(input_shape, n_filters, kernel_size):
    if len(input_shape) > 3: 
        depth_filter = input_shape[1]
    else:
        depth_filter = input_shape[0]
    
    learnable_params = (kernel_size**2 * n_filters +1) * depth_filter  # slightly different respect Conv2D classic since the bias respect the input channels
    print("The number of learnable parameters is -> {}".format(learnable_params))
    return learnable_params 

# 2D Pooling layer
def pool2d_out_shape(input_shape, kernel_size, stride):
    print("Dimensions after 2D pooling layer application, with filter size: {}x{}, and stride: {}\n".format(kernel_size, kernel_size, stride))
    if len(input_shape) > 3:  
        # batch case
        c_out = input_shape[1]
        print("The input shape is ->\t\t[batch: {}, channels: {}, height: {}, width: {}]".format(input_shape[0], str(c_out), str(input_shape[2]), str(input_shape[3])))
        h_out = int((math.floor(input_shape[2] - kernel_size)/stride) + 1)
        w_out = int((math.floor(input_shape[3] - kernel_size)/stride) + 1)
        print("The output shape is ->\t\t[batch: {}, channels: {}, height: {}, width: {}]".format(str(input_shape[0]), str(c_out), str(h_out), str(w_out)))
        return (input_shape[0], c_out, h_out, w_out)
        
    else:
        # single image case     
        c_out = input_shape[0]         
        print("The input shape is -> \t\t[channels: {}, height: {}, width: {}]".format(str(c_out), str(input_shape[1]), str(input_shape[2])))
        h_out = int((math.floor(input_shape[1] - kernel_size)/stride) + 1)
        w_out = int((math.floor(input_shape[2] - kernel_size)/stride) + 1)
        print("The output shape is ->\t\t[channels: {}, height: {}, width: {}]".format(str(c_out), str(h_out), str(w_out)))
        return (c_out, h_out, w_out)

""" 
    how to use dimension checkers: 
    conv2d_shapes(input_shape = (1,5,5), n_filters = 1, kernel_size= 3, padding = 0, stride=2)
    convTranspose2d_shapes(input_shape=(1,2,2), n_filters=1, kernel_size=2, padding=0, stride=2)
"""


# custom modules/functions
class expand_encoding(T.nn.Module):
    """ expand the encoding vector using the specified shape"""
    
    def __init__(self, shape = (-1, 2048, 1, 1)):
        super(expand_encoding,self).__init__()
        self.shape = shape
        
    def forward(self, x):
        return x.view(*self.shape)

class ExpLogger(object):
    """ Simple class to keep track of learning and experiment information"""
    
    def __init__(self, path_model):
        """ constructtor ExpLogger

        Args:
            path_model (str): path to model folder
        """
        super(ExpLogger, self).__init__()
        self.file_name      = "train.log"
        self.path_model     = path_model
        self.path_save      = os.path.join(self.path_model, self.file_name)
        self.delimiter_len  = 33
        self.delimiter_name = lambda x: "#"*self.delimiter_len +"{:^35}".format(x) + "#"*self.delimiter_len
        self.delimiter_line = "#"*(self.delimiter_len*2 + 35)
        self.train_lines    = []   # one line for each epoch
        self.start_time     = time()
        self.open_logger()
        
    def write_hyper(self, hyper_dict, name_section = "Hyperparameters"):
        self.write_logger(self.delimiter_name(name_section))
        text = "\n".join(["{:<40}: {:<40}".format(str(key),str(value)) for key, value in hyper_dict.items()])
        self.write_logger(text)
        self.write_logger(self.delimiter_line)
        self.flush()
        
    def write_config(self, config_dict, name_section = "Configuration"):
        self.write_logger(self.delimiter_name(name_section))
        text = "\n".join(["{:<40}: {:<40}".format(str(key),str(value)) for key, value in config_dict.items()])
        self.write_logger(text)
        self.write_logger(self.delimiter_line)
        self.flush()
        
    # def write_custom_section(self, custom_dict):
        
    def write_model(self, model_summary: str, name_section = "Model architecture"):
        self.write_logger(self.delimiter_name(name_section))
        self.write_logger(model_summary)
        self.write_logger(self.delimiter_line)
        self.flush()
    
    def log(self, epoch_dict: dict, in_line = True):
        """ log the current epoch

        Args:
            epoch_dict (dict): dictionary describing info from the current epoch, can include whatever information
        """
        if in_line:
            text = ""
            for key, value in epoch_dict.items():
                if key == "epoch":
                    text += "{}: {:<3}".format(str(key),str(value)) + " "
                else:
                    text += "{}: {:<15}".format(str(key),str(value)) + " "
            
            
            # text = "\t".join(["{}: {:<7}".format(str(key),str(value)) for key, value in epoch_dict.items()])
        else:
            text = "\n".join(["{:<20}: {:<10}".format(str(key),str(value)) for key, value in epoch_dict.items()]) + "\n"
        if self.train_lines == []:
            self.write_logger(self.delimiter_name("Training"))
        self.write_logger(text)
        self.train_lines.append(text)
    
    def log_mem(self, mem_summ):
        # self.write_logger(self.delimiter_line)
        self.write_logger(self.delimiter_line)
        self.write_logger(self.delimiter_name("Memory usage"))
        self.write_logger(mem_summ)
        self.write_logger(self.delimiter_line)
        self.flush()
    
    def end_log(self):
        # end saving information about training duration
        self.write_logger("\nTotal model training time: "+ self.get_TrainingTime())
        self.close_logger()
     
    def open_logger(self):
        if os.path.exists(self.path_save):
            os.remove(self.path_save)
        self.file = open(self.path_save, "a")
        self.file.write("*** Started model training log ***\n\n\n\n")
    
    def write_logger(self, text):
        self.file.write(text + "\n")
    
    def flush(self):
        self.file.flush()
    
    def get_TrainingTime(self):
        """
            simple function that returns the time elapsed from the Logger instantiation 
            and the ending of logging (training time), in format %h:%m:%s
        """
        
        train_time = time() - self.start_time
        
        # compute hours, minuts and rest seconds
        hours, rest_seconds = divmod(train_time, 3600)
        minutes, seconds    = divmod(rest_seconds, 60)

        # Format the result
        formatted_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
        
        print(f"Logger: training time -> {formatted_time}")
        
        return formatted_time
        
    def close_logger(self):
        self.write_logger("\n\n\n\n*** Ended model training log ***")
        self.flush()
        self.file.close()
    
##################################################  General Python utilities ##########################################################

concat_dict = lambda x,y: {**x, **y}

def print_dict(x):
    for idx,(k,v) in enumerate(x.items()):
        print("{:>3}) {:<15} -> {:<15}".format(str(idx),str(k),str(v)))

def print_list(x):
    for idx, elem in enumerate(x): 
        print("{:>3}){}".format(str(idx), str(elem)))

def isFolder(path):
    """
    simple check if a path (relative or absolute) is related to a folder (returns True) or a file (returns False).
    This method also considers the hidden file.
    """
    name_file = path.split("/")[-1]
        
    match_file = re.match(r"^[^.]+\..*$", name_file)
    if match_file is None: return True
    else: return False
      
def check_folder(path:str, force = False):
    """ 
    check if a folder exists, otherwise create it. 
    if "force" is set to False (Default), this method has no return
    if "force" is set to True, the folder is always copied. In this case return the new path considering homonymous folders.
    """
    
    # remove slash if last character
    if path[-1] == "/": path = path[:-1]
    
    if(isFolder(path)):
        if(not os.path.exists(path)):
            os.makedirs(path)
            return
            
        # if forced try to save a copy even if file already exist
        elif force:
            path_folders = path.split("/")
            path_to = "/".join(path_folders[:-1])
            last_folder_name = path_folders[-1]
            print(path_to)
            print(last_folder_name)
            
            counter_copy = 0
            for file in os.listdir(path_to):
                if last_folder_name in file and (isFolder(os.path.join(path_to, file))):
                    counter_copy += 1 
            
            new_path = os.path.join(path_to, last_folder_name + "_" + str(counter_copy))
            
            os.makedirs(new_path)
            return new_path
        # else:
        #     print('The folder to create already exists, set the force parameter to "True" for a new version')
    else:
        raise ValueError("Impossible to create a folder, the path {} is relative to a file!".format(path))
    
