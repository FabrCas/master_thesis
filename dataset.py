import os
from PIL import Image
import numpy as np


from utilities import * 

# torch import
import torch as T
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
T.manual_seed(22)


CDDB_PATH       = "./data/CDDB"
CIFAR100_PATH   = "./data/cifar100"


#                           [Deepfake classification]

class CDDB_binary(Dataset):
    def __init__(self, width_img= 224, height_img = 224, train = True):
        super(CDDB_binary,self).__init__()
        self.train = train                      # boolean flag to select train or test data
        self.width_img = width_img
        self.height_img = height_img
        self.transform_ops = transforms.Compose([
            transforms.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
            transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
        ])
        
        # load the data paths and labels
        self.x_train = None; self.y_train = None; self.x_test = None; self.y_test = None
        self._scanData()
        

        

    def _scanData(self, real_folder = "0_real", fake_folder = "1_fake"):
        
        # initialize empty lists
        xs = []  # images path
        ys = []  # binary label for each image
        
        
        if self.train: print("looking for train data in: {}".format(CDDB_PATH))
        else: print("looking for test data in: {}".format(CDDB_PATH))
        
        data_models = sorted(os.listdir(CDDB_PATH))
        for data_model in data_models:                                      # loop over folders of each model
            
            if self.train:
                path_set_model = os.path.join(CDDB_PATH,data_model,"train")      # around 70% of all the data
            else:
                path_set_model = os.path.join(CDDB_PATH,data_model,"val")        # around 30% of all the data
            
            #                               extract for the selected set
            sub_dir_model = sorted(os.listdir(path_set_model))
            # print(sub_dir_model)
            
            # count how many samples you collect
            n_train     = 0
            n_test      = 0
            
            if not (real_folder in sub_dir_model and fake_folder in sub_dir_model):   # contains sub-categories
                for category in sub_dir_model:
                    path_category =  os.path.join(path_set_model, category)
                    path_category_real = os.path.join(path_category, real_folder)
                    path_category_fake = os.path.join(path_category, fake_folder)
                    # print(path_category_real, "\n", path_category_fake)
                    
                    # get local data
                    x_category_real = [os.path.join(path_category_real, name)for name in os.listdir(path_category_real)]
                    x_category_fake = [os.path.join(path_category_fake, name)for name in os.listdir(path_category_fake)]
                    y_category_real = [0]*len(x_category_real)
                    y_category_fake = [1]*len(x_category_fake)
                    
                    # save in global data
                    xs = [*xs, *x_category_real, *x_category_fake]
                    ys = [*ys, *y_category_real, *y_category_fake]
                    
                    if self.train: 
                        n_train = len(x_category_real) + len(x_category_fake)
                    else:
                        n_test  = len(x_category_real) + len(x_category_fake)
                    
                    
                    
            else:                                                               # 2 folder: "0_real", "1_fake"
                path_real = os.path.join(path_set_model, real_folder)
                path_fake = os.path.join(path_set_model, fake_folder)
                # print(path_real,"\n", path_fake)
                
                # get local data
                x_model_real = [os.path.join(path_real, name)for name in os.listdir(path_real)]
                x_model_fake = [os.path.join(path_fake, name)for name in os.listdir(path_fake)]
                y_model_real = [0]*len(x_model_real)
                y_model_fake = [1]*len(x_model_fake)
                
                # save in global data
                xs = [*xs, *x_model_real, *x_model_fake]
                ys = [*ys, *y_model_real, *y_model_fake]
                
                if self.train:
                    n_train = len(x_model_real) + len(x_model_fake)
                else:
                    n_test  = len(x_model_real) + len(x_model_fake)    
                
            if self.train: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_train))
            else: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_test))
            
       
        # load the correct data 
        if self.train:
            print("train samples: {:<10}".format(len(xs)))  
        else:
            print("test samples: {:<10}".format(len(xs)))
            
        self.x = xs
        self.y = ys    # 0-> real, 1 -> fake
        
    def _transform(self,x):
        return self.transform_ops(x)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img_path = self.x[idx]
        # print(img_path)
        img = Image.open(img_path)
        img = self._transform(img)
        
        # check whether grayscale image, perform pseudocolor inversion 
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)

        label = self.y[idx]
        
        # binary encoding to compute BCE (one-hot)
        label_vector = [0,0]
        label_vector[label] = 1
        label_vector = T.tensor(label_vector)     
        
        return img, label_vector



#TODO
class CDDB(Dataset):
    def __init__(self, width_img= 224, height_img = 224):
        super(CDDB,self).__init__()
        
        self.width_img = width_img
        self.height_img = height_img
        
        self.transform_ops = transforms.Compose([
            transforms.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
            transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
        ])

    def _transform(self,x):
        x_transformed = self.transform_ops(x)
    
    def __len__(self):
        # return len()
        return 0
    
    def __getitem__(self, idx):
        return None


#                           [OOD detection]

def getCIFAR100_dataset(train, width_img= 224, height_img = 224):
    """ get CIFAR100 dataset

    Args:
        train (bool): choose between the train or test set. Defaults to True.
        width_img (int, optional): img width for the resize. Defaults to 224.
        height_img (int, optional): img height for the resize. Defaults to 224.
        
    Returns:
        torch.Dataset : Cifar100 dataset object
    """
    transform_ops = transforms.Compose([
        transforms.Resize((width_img, height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
        transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
    ])
    
    download_files = False
    # create folder if not exists
    if not(os.path.exists(CIFAR100_PATH)):
        os.mkdir(CIFAR100_PATH)
        download_files = True
    
    # load cifar data
    if train:
        cifar100 = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=True, download=download_files, transform=transform_ops)
    else:
        cifar100 = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=False, download=download_files, transform=transform_ops)
    
    return cifar100
    
    
        
        
        
# [test section]
if __name__ == "__main__":
    
    def test_cddbinary():
        dataset = CDDB_binary(train= True)
        # test Dataset item get
        x,y = dataset.__getitem__(0)
        print(x.shape)
        print(y)
        showImage(x, name = "CDDB sample")
        

        from torch.utils.data import DataLoader
        
        # test Dataloader from Dataset
        dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle= True, drop_last= True, pin_memory= True)
        show = True
        for i,(x,y) in enumerate(dataloader):
            print(x.shape)
            print(y)
            if show: 
                for i in range(x.shape[0]):
                    img = x[i]
                    label = y[i]
                    if label[0] == 1: # real
                        name  = "real image"
                    else:
                        name  = "fake image"
                    showImage(img, name = name)
            break
    
    ds = getCIFAR100_dataset(train = False)
    train_loader = DataLoader(ds, batch_size=32, shuffle=True)
    print(len(train_loader))
    
    
    
    
    