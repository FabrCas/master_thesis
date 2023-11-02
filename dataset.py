import os
from PIL                                import Image
import numpy                            as np
import math
import random
from utilities                          import showImage, sampleValidSet
import torch                            as T
import torchvision
from torchvision                        import transforms
from torchvision.transforms.functional  import InterpolationMode
from torch.utils.data                   import Dataset, DataLoader
T.manual_seed(22)
random.seed(22)

# Paths and data structure
CDDB_PATH       = "./data/CDDB"
CIFAR100_PATH   = "./data/cifar100"

DF_GROUP_SUBJECT    = {
                        "virtual_environment":["crn", "imle"],
                        # "faces": ["deepfake", "glow/black_hair", "glow/blond_hair", "glow/brown_hair",
                        #           "stargan_gf/black_hair", "stargan_gf/blond_hair","stargan_gf/brown_hair",
                        #           "whichfaceisreal", "wild"],
                        "faces":    ["deepfake","glow","stargan_gf","whichfaceisreal", "wild"],
                        "apple":    ["cyclegan/apple"],
                        "horse":    ["cyclegan/horse"],
                        "orange":   ["cyclegan/orange"],
                        "summer":   ["cyclegan/summer"],
                        "winter":   ["cyclegan/winger"],
                        "zebra":    ["cyclegan/zebra"],
                        "bedroom":  ["stylegan/bedroom"],
                        "car":      ["stylegan/car"],
                        "cat":      ["stylegan/cat"],
                        "mix":      ["biggan","gaugan","san"],
                       }

DF_GROUP_CLASSES    = {
                        "deepfake sources": ["biggan","cyclegan","gaugan","stargan_gf","stylegan"],
                        "non-deepfake sources": ["glow", "crn", "imle", "san", "deepfake"],
                        "unknown models":["whichfaceisreal", "wild"]
                    }


##################################################### [Binary Deepfake classification] ################################################################

class CDDB_binary(Dataset):
    """_
        Dataset class that uses the full data from CDDB dataset as IN-distribuion for binary deepfake detection
    """
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
        # self.x_train = None; self.y_train = None;    # validation set is build using utilities module
        # self.x_test = None; self.y_test = None
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


class CDDB_binary_Partial(Dataset):
    """_
        Dataset class that uses the partial data from CDDB dataset for binary deepfake detection.
        Selecting the study case ("easy", "medium", "hard") the data are orgnized,
        using remaining samples as OOD data
    """
    def __init__(self, scenario, width_img= 224, height_img = 224, train = True, ood = False, augment = False):
        """_summary_

        Args:
            scenario (str): select between "easy","mid","hard" scenarios
            width_img (int, optional): image width reshape. Defaults to 224.
            height_img (int, optional): image height reshape. Defaults to 224.
            train (bool, optional): boolean flag to retrieve trainset, otherwise testset. Defaults to True.
            ood (bool, optional):   boolean flag to retrieve ID data, otherwise OOD. Defaults to False.
            augment (bool, optional):   boolean flag to activate the data augmentation. Defaults to False.
        """
        super(CDDB_binary_Partial,self).__init__()
        
        # boolean flags for data to return
        self.scenario   = scenario
        self.train      = train
        self.ood        = ood
        self.augment    = augment
        
             
        self.width_img = width_img
        self.height_img = height_img
        self.transform_ops = transforms.Compose([
            transforms.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
            transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
        ])
        
        # load the data paths and labels
        # self.x_train = None; self.y_train = None;    # validation set is build using utilities module
        # self.x_test = None; self.y_test = None
        
        # self.x_ood = None; self.y_ood = None;    # validation set is build using utilities module
        # self.x_test = None; self.y_test = None
        
    
        if   scenario.lower().strip() == "easy":
            self._scanData_easy()
        elif scenario.lower().strip() == "mid":
            self._scanData_mid()
        elif scenario.lower().strip() == "hard":
            self._scanData_hard()
        else:
            raise ValueError("wrong selection for the scenario parameter, choose between: easy,mid,hard")
        

    def _scanData_easy(self, real_folder = "0_real", fake_folder = "1_fake"):
        """
            use face img as ID, the rest is OOD 
        """
        
        # initialize empty lists
        xs = []  # images path
        ys = []  # binary label for each image
        
        if self.train: print("looking for train data in: {}".format(CDDB_PATH))
        else: print("looking for test data in: {}".format(CDDB_PATH))
        
        data_models = sorted(os.listdir(CDDB_PATH))
        
        # separate ID and OOD
        ID_groups = DF_GROUP_SUBJECT["faces"]
        data_models_ID = []; data_models_OOD = []
        for data_model_name in data_models:
            if data_model_name in ID_groups:
                data_models_ID.append(data_model_name)
            else:
                data_models_OOD.append(data_model_name)
        
        if self.ood == False:
            data_models = data_models_ID
        else:
            data_models = data_models_OOD
            
        print(data_models_ID)   
        print(data_models_OOD) 
                
        
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
    
    
    def _scanData_mid(self, real_folder = "0_real", fake_folder = "1_fake"):
        
        # initialize empty lists
        xs = []  # images path
        ys = []  # binary label for each image
        
        
        if self.train: print("looking for train data in: {}".format(CDDB_PATH))
        else: print("looking for test data in: {}".format(CDDB_PATH))
        
        data_models = sorted(os.listdir(CDDB_PATH))
        
        # separate ID and OOD
        ID_groups = [*DF_GROUP_CLASSES["deepfake sources"], *DF_GROUP_CLASSES["non-deepfake sources"]]
        print(ID_groups)
        data_models_ID = []; data_models_OOD = []
        for data_model_name in data_models:
            if data_model_name in ID_groups:
                data_models_ID.append(data_model_name)
            else:
                data_models_OOD.append(data_model_name)
        
        if self.ood == False:
            data_models = data_models_ID
        else:
            data_models = data_models_OOD
            
        print(data_models_ID)   
        print(data_models_OOD) 
        
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
        
    def _scanData_hard(self, real_folder = "0_real", fake_folder = "1_fake"):
        
        # initialize empty lists
        xs = []  # images path
        ys = []  # binary label for each image
        
        
        
        if self.train: print("looking for train data in: {}".format(CDDB_PATH))
        else: print("looking for test data in: {}".format(CDDB_PATH))
        
        data_models = sorted(os.listdir(CDDB_PATH))
        
        # separate 
        
        print(data_models)
        
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

##################################################### [Multi-Class Deepfake classification] ############################################################
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


##################################################### [Out-Of-Distribution Detection] ################################################################

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
        transforms.Resize((width_img, height_img), interpolation= InterpolationMode.BICUBIC, antialias= True),
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


class OOD_dataset(Dataset):
    def __init__(self, id_data, ood_data, balancing_mode:str = None, exact_samples:int = None):  # balancing_mode = "max","exact" or None
        super(OOD_dataset, self).__init__()
        
        assert isinstance(id_data, Dataset)
        assert isinstance(ood_data, Dataset)
        
        self.id_data        = id_data
        self.ood_data       = ood_data
        self.balancing_mode  = balancing_mode
        self.exact_samples = exact_samples
        self._count_samples()

    
    def _count_samples(self):
        print("Building OOD detection dataset...\nID samples: {}, OOD samples: {}, balancing mode: {}".format(len(self.id_data), len(self.ood_data), self.balancing_mode))
        
        if self.balancing_mode == "max":
            max_samples = min(len(self.id_data), len(self.ood_data))
            self.n_IDsamples    = max_samples
            self.n_OODsamples   = max_samples
            self.n_samples      = max_samples*2
            
            # compute indices
            self.id_indices  =  random.sample(range(len(self.id_data)), max_samples)
            self.ood_indices =  random.sample(range(len(self.ood_data)), max_samples)
            
        elif self.balancing_mode == "exact" and not(self.exact_samples is None):
            self.n_IDsamples    = math.floor(self.exact_samples/2)
            self.n_OODsamples   = math.floor(self.exact_samples/2)
            if self.exact_samples%2 == 1:
                self.n_IDsamples += 1
            self.n_samples      = self.exact_samples
            
            # compute indices
            self.id_indices  =  random.sample(range(len(self.id_data)), self.n_IDsamples)
            self.ood_indices =  random.sample(range(len(self.ood_data)), self.n_OODsamples)
            
        else:
            self.n_IDsamples    = len(self.id_data)
            self.n_OODsamples   = len(self.ood_data)
            self.n_samples      = self.n_IDsamples + self.n_OODsamples # be careful, don't use too different dataset in size
            
            self.id_indices = None
            self.ood_indices = None
            
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self,idx):
        if idx < self.n_IDsamples:
            if self.id_indices is None:   # sample by providen index, this is used when no balancing is appleid on data (all data ID and ODD returned)
                x, _ = self.id_data[idx]
            else:
                x, _ = self.id_data[self.id_indices[idx]]    # sample from random sampled index, this is used when balancing is appleid on data (max or exact mode)
            y = 0
        else:
            idx_ood = idx - self.n_IDsamples  # compute the index for the ood data
            if self.ood_indices is None:
                x, _ = self.ood_data[idx_ood]
            else:
                x, _ = self.ood_data[self.ood_indices[idx_ood]]
            y = 1
    
        # check whether grayscale image, perform pseudocolor inversion 
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        
        # label to encoding
        y_vector    = [0,0]             # zero vector
        y_vector[y] = 1                 # mark with one the correct position: [1,0] -> ID, [0,1]-> OOD
        y_vector = T.tensor(y_vector) 
        
        return x, y_vector
        
        
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
    
    def test_splitvalidation():
                
        from torch.utils.data import random_split
        from torch.utils.data import ConcatDataset
        
        # this implemented as a method is under the name sampleValidSet in utilties module
        
        train = CDDB_binary(train= True)
        test  = CDDB_binary(train = False)
        generator = T.Generator().manual_seed(22)
        
        print("train length", len(train))
        print("test length", len(test))
        all_data = len(train) + len(test)
        print("total samples n°", all_data)
        
        print("Data percentage distribution over sets before partition:")
        print("TrainSet [%]",round(100*(len(train)/all_data),2) )
        print("TestSet  [%]",round(100*(len(test)/all_data),2),"\n")
        
        
        if False: 
            """
                split data with the following strategy, validation set is the 10% of all data.
                These samples are extract half from training set and half from test set.
                after this we have almost the following distribution:
                training        -> 65%
                validation      -> 10%
                testing         -> 25% 
            """
            
            
            # compute relative percentage
            perc_train = round(((0.1 * all_data)*0.5/len(train)),3)
            perc_test = round(((0.1 * all_data)*0.5/len(test)),3)

            
            
            train, val_p1  = random_split(train,  [1-perc_train, perc_train], generator = generator)  #[0.92, 0.08]
            print(f"splitting train (- {perc_train}%) ->",len(val_p1), len(train))

            test,  val_p2  = random_split(test,  [1-perc_test, perc_test], generator = generator)   #[0.84, 0.16]
            print(f"splitting test (- {perc_test}%) ->",len(val_p2), len(test))
            
            val = ConcatDataset([val_p1, val_p2])
            print("validation length ->", len(val))
            
        else:
            """
                split data with the following strategy, validation set is the 10% of all data.
                These samples are extract all from the test set.
            """
            
            
            perc_test = round(((0.1 * all_data)/len(test)),3)
            
            test,  val  = random_split(test,  [1-perc_test, perc_test], generator = generator)
            print(f"splitting test (- {perc_test}%) ->",len(val), len(test))
            print("validation length", len(val))
            
        print("\nData percentage distribution over sets after partition:")
        print("TrainSet [%]",round(100*(len(train)/all_data),2) )
        print("TestSet  [%]",round(100*(len(test)/all_data),2)  )
        print("ValidSet [%]",round(100*(len(val)/all_data),2)   )
            
        
        # from subset you can easily get the dataloader in the usual way pass to dataloader
        # dl = T.utils.data.DataLoader(val1[1])
        # print(type(dl))
    
    def test_getValid():
        train = CDDB_binary(train= True)
        test  = CDDB_binary(train= False)
        
        train, valid, test = sampleValidSet(train, test, useTestSet= True, verbose= True)
        print(len(train))
        print(len(valid))
        print(len(test))
     
    def test_cifar():
        ds = getCIFAR100_dataset(train = False)
        train_loader = DataLoader(ds, batch_size=32, shuffle=True)
        train_loader = iter(train_loader)
        batch1 =  next(train_loader)
        batch2 = next(train_loader)
        showImage(batch1[0][31])
        
    def test_ood():
        cddb_dataset = CDDB_binary(train = True)
        cifar_dataset = getCIFAR100_dataset(train = True)

        dataset = OOD_dataset(cddb_dataset, cifar_dataset, balancing_mode="exact", exact_samples= 50)
        print(len(dataset))
        sample = dataset[25]
        img = sample[0]
        showImage(img)
        print(sample[1])
    
    def test_partial_bin_cddb():
        data = CDDB_binary_Partial(scenario="mid", ood=True, train= True)
    
    # pass
    # test_ood()
    test_partial_bin_cddb()
    
    
    
    