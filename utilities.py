import  os 
import  json
from    time                                import time
import  multiprocessing                     as mp
import  numpy                               as np
from    PIL                                 import Image
import  matplotlib.pyplot                   as plt
import  torch                               as T
from    torch.utils.data                    import DataLoader
from    tqdm                                import tqdm
from    torchvision                         import transforms
from    torchvision.transforms.functional   import InterpolationMode


#  binary classification metrics
from    sklearn.metrics     import precision_score, recall_score, f1_score, confusion_matrix, hamming_loss, jaccard_score, accuracy_score
#  multi-class classification metrics
from    sklearn.metrics     import auc, roc_curve, average_precision_score, precision_recall_curve


##################################################  image transformation/data augmentation ############################################

def transfResnet50(isTensor = False):
    """ function that returns trasnformation operations sequence for the image input to be compatible for ResNet50 model

    Returns:
        compose pytorch object
    """
    if isTensor:
        transform_ops = transforms.Resize((224, 224), interpolation= InterpolationMode.BILINEAR, antialias= True),
    
    else: 
        transform_ops = transforms.Compose([
            transforms.Resize((224, 224), interpolation= InterpolationMode.BILINEAR, antialias= True),
            transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
        ])
    
    return transform_ops

#TODO implement normalizer for learning 

class NormalizeByChannelMeanStd(T.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, T.Tensor):
            mean = T.tensor(mean)
        if not isinstance(std, T.Tensor):
            std = T.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1, so: [batch_size, color_channel, height, width]
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def add_noise(batch_input, complexity=0.5):
    return batch_input + np.random.normal(size=batch_input.shape, scale=1e-9 + complexity)

def add_distortion_noise(batch_input):
    distortion = np.random.uniform(low=0.9, high=1.2)
    return batch_input + np.random.normal(size=batch_input.shape, scale=1e-9 + distortion)
    

##################################################  Save/Load functions ###############################################################

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
        plt.imshow(img)
        plt.show()
    else:
        print("img data is not valid for the printing")
        
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

##################################################  Plot/show functions ###############################################################

def plot_loss(loss_array, title_plot = None, path_save = None, duration_timer = 2500):
    """ save and plot the loss by epochs

    Args:
        loss_array (list): list of avg loss for each epoch
        title_plot (str, optional): _title to exhibit on the plot
        path_save (str, optional): relative path for the location of the save folder
        duration_timer (int, optional): milliseconds used to show the plot 
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
    
    plt.show()

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
    
    # initialize timer to close plot
    if duration_timer is not None: 
        timer = fig.canvas.new_timer(interval = duration_timer) # timer object with time interval in ms
        timer.add_callback(close_event)
    
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
        
    if duration_timer is not None: timer.start()
    plt.show()
    
def plot_ROC_curve(fpr, tpr, path_save = None, duration_timer = 2500):
    def close_event():
        plt.close()
    
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

def plot_PR_curve(recalls, precisions, path_save = None, duration_timer = 2500):
    
    def close_event():
        plt.close()
    
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
        
    plt.show()

##################################################  Metrics functions #################################################################

def metrics_binClass(preds, targets, pred_probs, epoch_model="unknown", path_save = None, average = "macro"):
    
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
        
        # compute everage precision 
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
        "average_precision/":           avg_precision,                                                                        \
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
            print("Confusion matrix")
            print(v)
    
    
        # plot and save (if path specified) confusion matrix
    plot_cm(cm = metrics_results['confusion_matrix'], labels = ["real", "fake"], title_plot = None, path_save = path_save)
    
    # save the results (JSON file) if a path has been provided
    if path_save is not None:
        # metrics_results_ = metrics_results.copy()
        # metrics_results_['confusion_matrix'] = metrics_results_['confusion_matrix'].tolist()
        saveJson(os.path.join(path_save, 'binaryMetrics_' + epoch_model + '.json'), metrics_results)
    
    metrics_results['confusion_matrix'] = metrics_results['confusion_matrix'].tolist()
    
    return metrics_results


def metrics_OOD(targets, pred_probs, pos_label = 1, path_save = None):
    
    fpr, tpr, _ = roc_curve(targets, pred_probs, pos_label=1,  drop_intermediate= False)
    # auroc = auc(fpr, tpr)
    
    fpr95   = fpr_at_95_tpr(pred_probs, targets)
    
    det_err, thr_err = detection_error(pred_probs, targets, pos_label= pos_label)
    
    metric_results = {
        # "auroc":            auroc,
        "fpr95":            fpr95,
        "detection_error":  det_err,
        "thr_de":           thr_err
        
    }
    
    # save the results (JSON file) if a path has been provided
    if path_save is not None:
        saveJson(os.path.join(path_save, 'metricsOOD.json'), metric_results)
    
    return metric_results

#TODO
def metrics_multiClass():
    pass

def fpr_at_95_tpr(preds, labels):
    '''
    Returns the false positive rate when the true positive rate is at minimum 95%.
    '''
    fpr, tpr, _ = roc_curve(labels, preds)
    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)

def detection_error(preds, labels, pos_label = 1):
    '''
    Return the misclassification probability when TPR is 95%.
    '''
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label= pos_label)

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
