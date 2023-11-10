import os
from PIL                                    import Image
import numpy                                as np
import math
import random
from utilities                              import showImage, sampleValidSet
import torch                                as T
import torchvision
from torchvision                            import transforms
from torchvision.transforms                 import v2    # new version for tranformation methods
from torchvision.transforms.functional      import InterpolationMode
from torch.utils.data                       import Dataset, DataLoader
T.manual_seed(22)
random.seed(22)

# Paths and data structure
CDDB_PATH       = "./data/CDDB"
CIFAR100_PATH   = "./data/cifar100"
MNIST_PATH      = "./data/MNIST"
FMNIST_PATH      = "./data/FashionMNIST"


# dictionary to explicit the content in each model folder of the dataset, DON'T MODIFY
DF_GROUP_CONTENT    = {
                        "virtual_environment":["crn", "imle"],
                        # "faces": ["deepfake", "glow/black_hair", "glow/blond_hair", "glow/brown_hair",
                        #           "stargan_gf/black_hair", "stargan_gf/blond_hair","stargan_gf/brown_hair",
                        #           "whichfaceisreal", "wild"],
                        "faces":    ["deepfake","glow","stargan_gf","whichfaceisreal", "wild"],
                        "apple":    ["cyclegan/apple"],
                        "horse":    ["cyclegan/horse"],
                        "orange":   ["cyclegan/orange"],
                        "summer":   ["cyclegan/summer"],
                        "winter":   ["cyclegan/winter"],
                        "zebra":    ["cyclegan/zebra"],
                        "bedroom":  ["stylegan/bedroom"],
                        "car":      ["stylegan/car"],
                        "cat":      ["stylegan/cat"],
                        "mix":      ["biggan","gaugan","san"],
                       }

# dictionary to separate the 3 main groups of the CDDB dataset, DON'T MODIFY
DF_GROUP_CLASSES    = {
                        "deepfake sources": ["biggan","cyclegan","gaugan","stargan_gf","stylegan"],  # GAN models
                        "non-deepfake sources": ["glow", "crn", "imle", "san", "deepfake"],          # NON-GAN models
                        "unknown models":["whichfaceisreal", "wild"]                                 
                    }


# dictionary which defines what content/class use in the different scenarios, this can be modified changing the scenarios' settings 
CATEGORIES_SCENARIOS_ID = {
                        "content": ["faces"],                                                               # DF_GROUP_CONTENT keys
                        "group":  ["deepfake sources", "non-deepfake sources"],                             # DF_GROUP_CLASSES keys
                        "mix": ["biggan","gaugan","stargan_gf", "deepfake", "glow", "crn","wild"]           # models In-Distribution names
                        }


##################################################### [Binary Deepfake classification] ################################################################

class CDDB_binary(Dataset):
    """_
        Dataset class that uses the full data from CDDB dataset as In-Distribuion (ID) for binary deepfake detection
    """
    def __init__(self, width_img= 224, height_img = 224, train = True, augment = False):
        """ CDDB_binary constructor

        Args:                
            - width_img (int, optional): image width reshape. Defaults to 224.
            - height_img (int, optional): image height reshape. Defaults to 224.
            - train (bool, optional): boolean flag to retrieve trainset, otherwise testset. Defaults to True.
            - ood (bool, optional):   boolean flag to retrieve ID data, otherwise OOD. Defaults to False.
            - augment (bool, optional):   boolean flag to activate the data augmentation. Defaults to False.
        """
        
        super(CDDB_binary,self).__init__()
        self.train = train                      # boolean flag to select train or test data
        self.augment  = False
        self.width_img = width_img
        self.height_img = height_img
        if self.augment:
            self.transform_ops = transforms.Compose([
                v2.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
                v2.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
                
                # v2.ToImage(),
                # v2.ToDtype(T.float32, scale=True),
                
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.1),
                v2.RandAugment(num_ops = 1, magnitude= 7, num_magnitude_bins= 51, interpolation = InterpolationMode.BILINEAR),
            ])
        else:
            self.transform_ops = transforms.Compose([
                v2.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
                v2.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
                # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])   # normlization between -1 and 1, using the whole range uniformly, formula: (pixel - mean)/std
            ])
        
        
        # initialization of path for the input images and the labels
        self.x = None
        self.y = None
        
        # scan the data   
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
                        n_train += len(x_category_real) + len(x_category_fake)
                    else:
                        n_test  += len(x_category_real) + len(x_category_fake)
                    
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
                    n_train += len(x_model_real) + len(x_model_fake)
                else:
                    n_test  += len(x_model_real) + len(x_model_fake)    
                       
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

# Used for ID-OOD study
class CDDB_binary_Partial(Dataset):
    """_
        Dataset class that uses the partial data from CDDB dataset for binary deepfake detection.
        Selecting the study case ("content","group","mix") the data are organized,
        using remaining samples as OOD data.
    """
    def __init__(self, scenario, width_img= 224, height_img = 224, train = True, ood = False, augment = False):
        """ CDDB_binary_Partial constructor

        Args:
            - scenario (str): modality division ID and OOD. select between "content","group","mix" scenarios:
                - content: data (real/fake for each model that contains a certain type of images) from a pseudo-category,
                chosen only samples with faces, OOD -> all other data that contains different subject from the one in ID.
                - group: ID -> assign wo data group from CDDB (deep-fake resources, non-deep-fake resources),
                OOD-> the remaining data group (unknown models)
                - mix: mix ID and ODD without maintaining the integrity of the CDDB groups, i.e take models samples from
                1st ,2nd,3rd groups and do the same for OOD without intersection.
                
            - width_img (int, optional): image width reshape. Defaults to 224.
            - height_img (int, optional): image height reshape. Defaults to 224.
            - train (bool, optional): boolean flag to retrieve trainset, otherwise testset. Defaults to True.
            - ood (bool, optional):   boolean flag to retrieve ID data, otherwise OOD. Defaults to False.
            - augment (bool, optional):   boolean flag to activate the data augmentation. Defaults to False.
        """
        super(CDDB_binary_Partial,self).__init__()
        
        # boolean flags for data to return
        self.scenario   = scenario.lower().strip()
        self.train      = train
        self.ood        = ood
        self.augment    = augment
        
             
        self.width_img = width_img
        self.height_img = height_img
        if self.augment:
            self.transform_ops = transforms.Compose([
                v2.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
                v2.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
                
                # v2.ToImage(),
                # v2.ToDtype(T.float32, scale=True),
                
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.1),
                v2.RandAugment(num_ops = 1, magnitude= 7, num_magnitude_bins= 51, interpolation = InterpolationMode.BILINEAR),
            ])
        else:
            self.transform_ops = transforms.Compose([
                v2.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
                v2.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
                # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])   # normlization between -1 and 1, using the whole range uniformly, formula: (pixel - mean)/std
            ])
        
        # initialization of path for the input images and the labels
        self.x = None
        self.y = None
        
        # scan the data    
        if  scenario in ["content","group","mix"]:
            self._scanData(scenario = scenario)
        else:
            raise ValueError("wrong selection for the scenario parameter, choose between: content, group and mix")
        

    def _scanData(self, scenario, real_folder = "0_real", fake_folder = "1_fake"):
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
        cateogories = CATEGORIES_SCENARIOS_ID[scenario]
            
        ID_groups   = []
        # handle content and group scenario
        if scenario == "content" or scenario == "group":
            for category in cateogories:
                if scenario == "content":
                    ID_groups = [*ID_groups, *DF_GROUP_CONTENT[category]]
                elif scenario == "group":
                    ID_groups = [*ID_groups, *DF_GROUP_CLASSES[category]]
                    
        # handle mix scenario  
        else:  
            ID_groups = cateogories

        # split OOD and ID models name
        data_models_ID = []; data_models_OOD = []
        for data_model_name in data_models:
            if data_model_name in ID_groups:
                data_models_ID.append(data_model_name)
            else:
                data_models_OOD.append(data_model_name)
        
        # set name models to scan 
        if self.ood == False:
            data_models = data_models_ID
        else:
            data_models = data_models_OOD 
        # print(data_models)    
        
    
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
                        n_train += len(x_category_real) + len(x_category_fake)
                    else:
                        n_test  += len(x_category_real) + len(x_category_fake)
                    
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
                    n_train += len(x_model_real) + len(x_model_fake)
                else:
                    n_test  += len(x_model_real) + len(x_model_fake)    
                
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
        img = self._transform(img)    #dtype = float32
        
        
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
class CDDB(Dataset):
    """_
        Dataset class that uses the full data from CDDB dataset as In-Distribuion (ID) for multi-label deepfake detection
    """
    def __init__(self, width_img= 224, height_img = 224, train = True, augment = False, real_grouping = "single"):
        """
        CDDB_binary_Partial constructor

        Args:                
            - width_img (int, optional): image width reshape. Defaults to 224.
            - height_img (int, optional): image height reshape. Defaults to 224.
            - train (bool, optional): boolean flag to retrieve trainset, otherwise testset. Defaults to True.
            - ood (bool, optional):   boolean flag to retrieve ID data, otherwise OOD. Defaults to False.
            - augment (bool, optional):   boolean flag to activate the data augmentation. Defaults to False.
            - real_grouping (str): values: "single", "categories" or "models". string used to choose the modality to group the real labels,
                - "single" means just one label for all the real images,
                - "category" is for different labels of real images contains content like faces, cars, cats,etc.
                - "models" separate the real images for each of the models present in the dataset. so real and fake labels have equal number
        """
        super(CDDB,self).__init__()
        self.train              = train                      # boolean flag to select train or test data
        self.augment            = augment
        self.real_grouping      = real_grouping
        self.width_img          = width_img
        self.height_img         = height_img
        if self.augment:
            self.transform_ops = transforms.Compose([
                v2.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
                v2.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
                
                # v2.ToImage(),
                # v2.ToDtype(T.float32, scale=True),
            
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.1),
                v2.RandAugment(num_ops = 1, magnitude= 7, num_magnitude_bins= 51, interpolation = InterpolationMode.BILINEAR),
            ])
        else:
            self.transform_ops = transforms.Compose([
                v2.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
                v2.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
                # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])   # normlization between -1 and 1, using the whole range uniformly, formula: (pixel - mean)/std
            ])
        
        
        # initialization of path for the input images and the labels
        self.x = None
        self.y = None
        
        # define the labels
        if self.real_grouping == "single":    # only one label for the real labels, needs downsample or usage of weights during training 
            self.idx2label = ['biggan', 'crn', 'cyclegan', 'deepfake', 'gaugan', 'glow', 'imle', 'san', 'stargan_gf', 'stylegan', 'whichfaceisreal',
                              'wild', 'real']
            # scan the data   
            self._scanData_singleReal()
        elif self.real_grouping == "categories":    # in this modality i not use real images folder with mixed content: i.e in biggan
            self.idx2label = ['biggan', 'crn', 'cyclegan', 'deepfake', 'gaugan', 'glow', 'imle', 'real_apple', 'real_bedroom', 'real_cat', 'real_car',
                              'real_faces', 'real_horse', 'real_orange', 'real_summer', 'real_virtual_environment', 'real_winter', 'real_zebra', 'san',
                              'stargan_gf', 'stylegan', 'whichfaceisreal', 'wild']
            
            # scan the data   
            self._scanData_categoriesReal()
        elif self.real_grouping == "models": # a real and fake class for each model (less useful?)
            self.idx2label = ['biggan_fake', 'biggan_real', 'crn_fake', 'crn_real', 'cyclegan_fake', 'cyclegan_real', 'deepfake_fake', 'deepfake_real',
                              'gaugan_fake', 'gaugan_real', 'glow_fake', 'glow_real', 'imle_fake', 'imle_real', 'san_fake', 'san_real', 'stargan_gf_fake',
                              'stargan_gf_real', 'stylegan_fake', 'stylegan_real', 'whichfaceisreal_fake', 'whichfaceisreal_real', 'wild_fake', 'wild_real']
            # scan the data   
            self._scanData_allReal()
            
    def _scanData_singleReal(self, real_folder = "0_real", fake_folder = "1_fake"):
        
        # initialize empty lists
        xs = []  # images path
        ys = []  # binary label for each image
        
        if self.train: print("looking for train data in: {}".format(CDDB_PATH))
        else: print("looking for test data in: {}".format(CDDB_PATH))
        
        data_models = sorted(os.listdir(CDDB_PATH))
        
        # take the integer value used to represent the real class
        idx_real_label = self.idx2label.index("real")
        
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
            
            
            # take the integer value used to represent the fake class for the current model
            idx_fake_label = self.idx2label.index(data_model)

            
            if not (real_folder in sub_dir_model and fake_folder in sub_dir_model):   # contains sub-categories
                for category in sub_dir_model:
                    path_category =  os.path.join(path_set_model, category)
                    path_category_real = os.path.join(path_category, real_folder)
                    path_category_fake = os.path.join(path_category, fake_folder)
                    # print(path_category_real, "\n", path_category_fake)
                    
                    # get local data
                    x_category_real = [os.path.join(path_category_real, name)for name in os.listdir(path_category_real)]
                    x_category_fake = [os.path.join(path_category_fake, name)for name in os.listdir(path_category_fake)]
                    y_category_real = [idx_real_label]*len(x_category_real)
                    y_category_fake = [idx_fake_label]*len(x_category_fake)
                    
                    # save in global data
                    xs = [*xs, *x_category_real, *x_category_fake]
                    ys = [*ys, *y_category_real, *y_category_fake]
                    
                    if self.train: 
                        n_train += len(x_category_real) + len(x_category_fake)
                    else:
                        n_test  += len(x_category_real) + len(x_category_fake)
                    
            else:                                                               # 2 folder: "0_real", "1_fake"
                path_real = os.path.join(path_set_model, real_folder)
                path_fake = os.path.join(path_set_model, fake_folder)
                # print(path_real,"\n", path_fake)
                
                # get local data
                x_model_real = [os.path.join(path_real, name)for name in os.listdir(path_real)]
                x_model_fake = [os.path.join(path_fake, name)for name in os.listdir(path_fake)]
                y_model_real = [idx_real_label]*len(x_model_real)
                y_model_fake = [idx_fake_label]*len(x_model_fake)
                
                # save in global data
                xs = [*xs, *x_model_real, *x_model_fake]
                ys = [*ys, *y_model_real, *y_model_fake]
                
                if self.train:
                    n_train += len(x_model_real) + len(x_model_fake)
                else:
                    n_test  += len(x_model_real) + len(x_model_fake)    
                
            if self.train: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_train))
            else: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_test))
            
       
        # load the correct data 
        if self.train:
            print("train samples: {:<10}".format(len(xs)))  
        else:
            print("test samples: {:<10}".format(len(xs)))
            
        self.x = xs
        self.y = ys    # 0-> real, 1 -> fake
    
    def _scanData_categoriesReal(self, real_folder = "0_real", fake_folder = "1_fake"):
        
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
            
            print("\n\t\t\t"+ data_model +"\n")
            # take the integer value used to represent the fake class for the current model
            idx_fake_label = self.idx2label.index(data_model)
            print("label ID fake sample -> ", data_model, idx_fake_label)
            
            if not (real_folder in sub_dir_model and fake_folder in sub_dir_model):   # contains sub-categories
                for category in sub_dir_model:
                    path_category =  os.path.join(path_set_model, category)
                    path_category_real = os.path.join(path_category, real_folder)
                    path_category_fake = os.path.join(path_category, fake_folder)
                    # print(path_category_real, "\n", path_category_fake)
                    
                    # get local data
                    full_name_category = data_model + "/" + category

                    # fake samples paths
                    x_category_fake = [os.path.join(path_category_fake, name)for name in os.listdir(path_category_fake)]
                    y_category_fake = [idx_fake_label]*len(x_category_fake)
                    
                    
                    # compute idx real label
                    idx_real_label = None
                    if "hair" in category:                # sub categories for the faces
                            idx_real_label = self.idx2label.index("real_faces")
                            print("label ID real sample -> real_faces", idx_real_label)
                    else:
                        for k,v in DF_GROUP_CONTENT.items():
                            if full_name_category in v:
                                if k == "mix": break
                                category = "real_" + k
                                idx_real_label = self.idx2label.index(category)
                                print("label ID real sample -> ", category,idx_real_label)
                                break
                   
                    
                    # real samples paths, skip samples if contains mixed content (not classifiable)
                    if not(data_model in DF_GROUP_CONTENT['mix']):
                        x_category_real = [os.path.join(path_category_real, name)for name in os.listdir(path_category_real)]
                        y_category_real = [idx_real_label]*len(x_category_real)
                    else:
                        x_category_real = []
                        y_category_real = []
                    
                    # save in global data
                    xs = [*xs, *x_category_real, *x_category_fake]
                    ys = [*ys, *y_category_real, *y_category_fake]
                    
                    if self.train: 
                        n_train += len(x_category_real) + len(x_category_fake)
                    else:
                        n_test  += len(x_category_real) + len(x_category_fake)
                    
            else:                                                               # 2 folder: "0_real", "1_fake"
                path_real = os.path.join(path_set_model, real_folder)
                path_fake = os.path.join(path_set_model, fake_folder)
                # print(path_real,"\n", path_fake)
                
                # get local data
                # fake samples paths
                x_model_fake = [os.path.join(path_fake, name)for name in os.listdir(path_fake)]
                y_model_fake = [idx_fake_label]*len(x_model_fake)
                
                # compute idx real label
                idx_real_label = None
                for k,v in DF_GROUP_CONTENT.items():
                    if data_model in v:
                        if k == "mix": break
                        category = "real_" + k
                        idx_real_label = self.idx2label.index(category)
                        print("label ID real sample -> ", category, idx_real_label)
                        break
                    
                
                
                 # real samples paths, skip samples if contains mixed content (not classifiable)
                if not(data_model in DF_GROUP_CONTENT['mix']):
                    x_model_real = [os.path.join(path_real, name)for name in os.listdir(path_real)]
                    y_model_real = [idx_real_label]*len(x_model_real)
                else:
                    x_model_real = []
                    y_model_real = []
                    
                
                # save in global data
                xs = [*xs, *x_model_real, *x_model_fake]
                ys = [*ys, *y_model_real, *y_model_fake]
                
                if self.train:
                    n_train += len(x_model_real) + len(x_model_fake)
                else:
                    n_test  += len(x_model_real) + len(x_model_fake)    
                
            if self.train: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_train))
            else: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_test))
            
       
        # print the total number of examples
        if self.train:
            print("train samples: {:<10}".format(len(xs)))  
        else:
            print("test samples: {:<10}".format(len(xs)))
            
        self.x = xs
        self.y = ys    # 0-> real, 1 -> fake
    
    def _scanData_allReal(self, real_folder = "0_real", fake_folder = "1_fake"):

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
            
            
            # take the integer value used to represent the fake and real class for the current model
            idx_fake_label = self.idx2label.index(data_model + "_fake")
            idx_real_label = self.idx2label.index(data_model + "_real")

            print(idx_fake_label)
            print(idx_real_label)
            
            if not (real_folder in sub_dir_model and fake_folder in sub_dir_model):   # contains sub-categories
                for category in sub_dir_model:
                    path_category =  os.path.join(path_set_model, category)
                    path_category_real = os.path.join(path_category, real_folder)
                    path_category_fake = os.path.join(path_category, fake_folder)
                    # print(path_category_real, "\n", path_category_fake)
                    
                    # get local data
                    x_category_real = [os.path.join(path_category_real, name)for name in os.listdir(path_category_real)]
                    x_category_fake = [os.path.join(path_category_fake, name)for name in os.listdir(path_category_fake)]
                    y_category_real = [idx_real_label]*len(x_category_real)
                    y_category_fake = [idx_fake_label]*len(x_category_fake)
                    
                    # save in global data
                    xs = [*xs, *x_category_real, *x_category_fake]
                    ys = [*ys, *y_category_real, *y_category_fake]
                    
                    if self.train: 
                        n_train += len(x_category_real) + len(x_category_fake)
                    else:
                        n_test  += len(x_category_real) + len(x_category_fake)
                    
            else:                                                               # 2 folder: "0_real", "1_fake"
                path_real = os.path.join(path_set_model, real_folder)
                path_fake = os.path.join(path_set_model, fake_folder)
                # print(path_real,"\n", path_fake)
                
                # get local data
                x_model_real = [os.path.join(path_real, name)for name in os.listdir(path_real)]
                x_model_fake = [os.path.join(path_fake, name)for name in os.listdir(path_fake)]
                y_model_real = [idx_real_label]*len(x_model_real)
                y_model_fake = [idx_fake_label]*len(x_model_fake)
                
                # save in global data
                xs = [*xs, *x_model_real, *x_model_fake]
                ys = [*ys, *y_model_real, *y_model_fake]
                
                if self.train:
                    n_train += len(x_model_real) + len(x_model_fake)
                else:
                    n_test  += len(x_model_real) + len(x_model_fake)    
                
            if self.train: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_train))
            else: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_test))
            
       
        # print the total number of examples
        if self.train:
            print("train samples: {:<10}".format(len(xs)))  
        else:
            print("test samples: {:<10}".format(len(xs)))
            
        self.x = xs
        self.y = ys    # 0-> real, 1 -> fake
    
    def _transform(self,x):
        return self.transform_ops(x)
    
    def _one_hot_encoding(self, label_idx):
        encoding = [0] * len(self.idx2label)
        encoding[label_idx] = 1
        return encoding
     
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img_path = self.x[idx]
        # print(img_path)
        img = Image.open(img_path)
        img = self._transform(img)    #dtype = float32
        
        
        # check whether grayscale image, perform pseudocolor inversion 
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)

        # sample the label
        label = self.y[idx]    # if it's necessary the encoding, use self._one_hot_encoding(self.y[idx])
        return img, label

# Used for ID-OOD study
class CDDB_Partial(Dataset):
    """_
        Dataset class that uses partial data from CDDB dataset as In-Distribuion (ID) for multi-label deepfake detection,
        Selecting the study case ("content","group","mix") the data are organized,
        using remaining samples as OOD data.
    """
    def __init__(self, scenario, width_img= 224, height_img = 224, train = True, ood = False, augment = False, real_grouping = "single"):
        """CDDB_Partial constructor

        Args:
            - scenario (str): modality division ID and OOD. select between "content","group","mix" scenarios:
                - content: data (real/fake for each model that contains a certain type of images) from a pseudo-category,
                chosen only samples with faces, OOD -> all other data that contains different subject from the one in ID.
                - group: ID -> assign wo data group from CDDB (deep-fake resources, non-deep-fake resources),
                OOD-> the remaining data group (unknown models)
                - mix: mix ID and ODD without maintaining the integrity of the CDDB groups, i.e take models samples from
                1st ,2nd,3rd groups and do the same for OOD without intersection.
                
            - width_img (int, optional): image width reshape. Defaults to 224.
            - height_img (int, optional): image height reshape. Defaults to 224.
            - train (bool, optional): boolean flag to retrieve trainset, otherwise testset. Defaults to True.
            - ood (bool, optional):   boolean flag to retrieve ID data, otherwise OOD. Defaults to False.
            - augment (bool, optional):   boolean flag to activate the data augmentation. Defaults to False.
            - real_grouping (str): values: "single", "categories" or "models". string used to choose the modality to group the real labels,
                - "single" means just one label for all the real images,
                - "category" is for different labels of real images contains content like faces, cars, cats,etc.
                - "models" separate the real images for each of the models present in the dataset. so real and fake labels have equal number
        """
        super(CDDB_Partial,self).__init__()
        self.scenario           = scenario
        self.train              = train                      # boolean flag to select train or test data
        self.augment            = augment
        self.ood                = ood
        self.real_grouping      = real_grouping
        self.width_img          = width_img
        self.height_img         = height_img
        if self.augment:
            self.transform_ops = transforms.Compose([
                v2.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
                v2.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
                
                # v2.ToImage(),
                # v2.ToDtype(T.float32, scale=True),
                
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.1),
                v2.RandAugment(num_ops = 1, magnitude= 7, num_magnitude_bins= 51, interpolation = InterpolationMode.BILINEAR),
            ])
        else:
            self.transform_ops = transforms.Compose([
                v2.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
                v2.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
                # transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])   # normlization between -1 and 1, using the whole range uniformly, formula: (pixel - mean)/std
            ])
        
        
        # initialization of path for the input images and the labels
        self.x = None
        self.y = None
        
        # define the labels
        # idx2label list is reduced base on the scenario selected, moreover the labels presented in the list are based on the ood flag value
        if self.real_grouping == "single":    # only one label for the real labels, needs downsample or usage of weights during training 
            self.idx2label = ['biggan', 'crn', 'cyclegan', 'deepfake', 'gaugan', 'glow', 'imle', 'san', 'stargan_gf', 'stylegan', 'whichfaceisreal',
                              'wild', 'real']
            # scan the data   
            self._scanData_singleReal()
        elif self.real_grouping == "categories":    # in this modality i not use real images folder with mixed content: i.e in biggan
            self.idx2label = ['biggan', 'crn', 'cyclegan', 'deepfake', 'gaugan', 'glow', 'imle', 'real_apple', 'real_bedroom', 'real_cat', 'real_car',
                              'real_faces', 'real_horse', 'real_orange', 'real_summer', 'real_virtual_environment', 'real_winter', 'real_zebra', 'san',
                              'stargan_gf', 'stylegan', 'whichfaceisreal', 'wild']
            
            # scan the data   
            self._scanData_categoriesReal()
        elif self.real_grouping == "models": # a real and fake class for each model (less useful?)
            self.idx2label = ['biggan_fake', 'biggan_real', 'crn_fake', 'crn_real', 'cyclegan_fake', 'cyclegan_real', 'deepfake_fake', 'deepfake_real',
                              'gaugan_fake', 'gaugan_real', 'glow_fake', 'glow_real', 'imle_fake', 'imle_real', 'san_fake', 'san_real', 'stargan_gf_fake',
                              'stargan_gf_real', 'stylegan_fake', 'stylegan_real', 'whichfaceisreal_fake', 'whichfaceisreal_real', 'wild_fake', 'wild_real']
            # scan the data   
            self._scanData_allReal()
            
    def _scanData_singleReal(self, real_folder = "0_real", fake_folder = "1_fake"):
        
        # initialize empty lists
        xs = []  # images path
        ys = []  # binary label for each image
        
        if self.train: print("looking for train data in: {}".format(CDDB_PATH))
        else: print("looking for test data in: {}".format(CDDB_PATH))
        
        data_models = sorted(os.listdir(CDDB_PATH))
        
        # separate ID and OOD
        cateogories = CATEGORIES_SCENARIOS_ID[self.scenario]
            
        ID_groups   = []
        # handle content and group scenario
        if self.scenario == "content" or self.scenario == "group":
            for category in cateogories:
                if self.scenario == "content":
                    ID_groups = [*ID_groups, *DF_GROUP_CONTENT[category]]
                elif self.scenario == "group":
                    ID_groups = [*ID_groups, *DF_GROUP_CLASSES[category]]
                    
        # handle mix scenario  
        else:  
            ID_groups = cateogories

        # get the model needed from the category is contains sub-directory, update In-Distribution group
        for idx,name in enumerate(ID_groups.copy()):
            if "/" in name:
                ID_groups[idx] = name.split("/")[0]   # take just the model
        
        # split OOD and ID models name
        data_models_ID = []; data_models_OOD = []
        for data_model_name in data_models:
            if data_model_name in ID_groups:
                data_models_ID.append(data_model_name)
            else:
                data_models_OOD.append(data_model_name)
        
        # set name models to scan 
        if self.ood == False:
            data_models = data_models_ID
        else:
            data_models = data_models_OOD 
        
        # shrink the idx2label list
        removed = 0
        for idx, label in enumerate(self.idx2label.copy()):
            if label in data_models or label == "real":
                continue
            else:
                self.idx2label.pop(idx - removed)
                removed += 1 
                
        # take the integer value used to represent the real class
        idx_real_label = self.idx2label.index("real")
        
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
            
            
            # take the integer value used to represent the fake class for the current model
            idx_fake_label = self.idx2label.index(data_model)

            
            if not (real_folder in sub_dir_model and fake_folder in sub_dir_model):   # contains sub-categories
                for category in sub_dir_model:
                    path_category =  os.path.join(path_set_model, category)
                    path_category_real = os.path.join(path_category, real_folder)
                    path_category_fake = os.path.join(path_category, fake_folder)
                    # print(path_category_real, "\n", path_category_fake)
                    
                    # get local data
                    x_category_real = [os.path.join(path_category_real, name)for name in os.listdir(path_category_real)]
                    x_category_fake = [os.path.join(path_category_fake, name)for name in os.listdir(path_category_fake)]
                    y_category_real = [idx_real_label]*len(x_category_real)
                    y_category_fake = [idx_fake_label]*len(x_category_fake)
                    
                    # save in global data
                    xs = [*xs, *x_category_real, *x_category_fake]
                    ys = [*ys, *y_category_real, *y_category_fake]
                    
                    if self.train: 
                        n_train += len(x_category_real) + len(x_category_fake)
                    else:
                        n_test  += len(x_category_real) + len(x_category_fake)
                    
            else:                                                               # 2 folder: "0_real", "1_fake"
                path_real = os.path.join(path_set_model, real_folder)
                path_fake = os.path.join(path_set_model, fake_folder)
                # print(path_real,"\n", path_fake)
                
                # get local data
                x_model_real = [os.path.join(path_real, name)for name in os.listdir(path_real)]
                x_model_fake = [os.path.join(path_fake, name)for name in os.listdir(path_fake)]
                y_model_real = [idx_real_label]*len(x_model_real)
                y_model_fake = [idx_fake_label]*len(x_model_fake)
                
                # save in global data
                xs = [*xs, *x_model_real, *x_model_fake]
                ys = [*ys, *y_model_real, *y_model_fake]
                
                if self.train:
                    n_train += len(x_model_real) + len(x_model_fake)
                else:
                    n_test  += len(x_model_real) + len(x_model_fake)    
                
            if self.train: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_train))
            else: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_test))
            
       
        # load the correct data 
        if self.train:
            print("train samples: {:<10}".format(len(xs)))  
        else:
            print("test samples: {:<10}".format(len(xs)))
            
        self.x = xs
        self.y = ys    # 0-> real, 1 -> fake
    
    def _scanData_categoriesReal(self, real_folder = "0_real", fake_folder = "1_fake", verbose = False):
        
        # initialize empty lists
        xs = []  # images path
        ys = []  # binary label for each image
        
        if self.train: print("looking for train data in: {}".format(CDDB_PATH))
        else: print("looking for test data in: {}".format(CDDB_PATH))
        
        data_models = sorted(os.listdir(CDDB_PATH))

        # separate ID and OOD
        cateogories = CATEGORIES_SCENARIOS_ID[self.scenario]
            
        ID_groups   = []
        # handle content and group scenario
        if self.scenario == "content" or self.scenario == "group":
            for category in cateogories:
                if self.scenario == "content":
                    ID_groups = [*ID_groups, *DF_GROUP_CONTENT[category]]
                elif self.scenario == "group":
                    ID_groups = [*ID_groups, *DF_GROUP_CLASSES[category]]
                    
        # handle mix scenario  
        else:  
            ID_groups = cateogories

        # get the model needed from the category is contains sub-directory, update In-Distribution group
        for idx,name in enumerate(ID_groups.copy()):
            if "/" in name:
                ID_groups[idx] = name.split("/")[0]   # take just the model
            
        if verbose: print("In distribution groupds ->", ID_groups)
        
        # split OOD and ID models name
        data_models_ID = []; data_models_OOD = []
        for data_model_name in data_models:
            if data_model_name in ID_groups:
                data_models_ID.append(data_model_name)
            else:
                data_models_OOD.append(data_model_name)
        
        # set name models to scan 
        if self.ood == False:
            data_models = data_models_ID
        else:
            data_models = data_models_OOD 
        
        if verbose: print("labels before shrinking -> ", self.idx2label)
        
        # shrink the idx2label list
        
        if self.scenario == "content":
            removed = 0
            for idx, label in enumerate(self.idx2label.copy()):
                # belong to a fake model
                if not "real_" in (label):
                    if label in data_models:
                        continue    #keep
                    else:
                        self.idx2label.pop(idx - removed)
                        removed += 1
                else: # belong to a real category
                    category = label.replace("real_", "")
                    models_in = [model_in.split("/")[0] for model_in in DF_GROUP_CONTENT[category]]
                    if verbose: print(category, " -> ", models_in)
                    if any(data_model in models_in for data_model in data_models) and category in CATEGORIES_SCENARIOS_ID[self.scenario]:
                        continue #keep
                    else:
                        self.idx2label.pop(idx - removed)
                        removed += 1
                        
        elif self.scenario == "group" or self.scenario == "mix":
            removed = 0
            for idx, label in enumerate(self.idx2label.copy()):
                # belong to a fake model
                if not "real_" in (label):
                    if label in data_models:
                        continue    #keep
                    else:
                        self.idx2label.pop(idx - removed)
                        removed += 1
                else: # belong to a real category
                    category = label.replace("real_", "")
                    models_in = [model_in.split("/")[0] for model_in in DF_GROUP_CONTENT[category]]
                    if verbose:print(category, " -> ", models_in)
                    if any(data_model in models_in for data_model in data_models):
                        continue #keep
                    else:
                        self.idx2label.pop(idx - removed)
                        removed += 1
  
        else:
            raise Exception
        
        if verbose:    
            print("labels: ", self.idx2label)
            print("models: ",data_models)
        
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
            
            print("\n\t\t\t"+ data_model +"\n")
            # take the integer value used to represent the fake class for the current model
            idx_fake_label = self.idx2label.index(data_model)
            print("label ID fake sample -> ", data_model, idx_fake_label)
            
            if not (real_folder in sub_dir_model and fake_folder in sub_dir_model):   # contains sub-categories
                for category in sub_dir_model:
                    path_category =  os.path.join(path_set_model, category)
                    path_category_real = os.path.join(path_category, real_folder)
                    path_category_fake = os.path.join(path_category, fake_folder)
                    # print(path_category_real, "\n", path_category_fake)
                    
                    # get local data
                    full_name_category = data_model + "/" + category

                    # fake samples paths
                    x_category_fake = [os.path.join(path_category_fake, name)for name in os.listdir(path_category_fake)]
                    y_category_fake = [idx_fake_label]*len(x_category_fake)
                    
                    
                    # compute idx real label
                    idx_real_label = None
                    try:
                        if "hair" in category and "faces" in CATEGORIES_SCENARIOS_ID["content"]:  # sub categories in the model forlder that are all under "faces" class
                                idx_real_label = self.idx2label.index("real_faces")
                                print("label ID real sample -> real_faces", idx_real_label)
                        else:
                            for k,v in DF_GROUP_CONTENT.items():
                                if full_name_category in v:
                                    if k == "mix": break
                                    category = "real_" + k
                                    idx_real_label = self.idx2label.index(category)
                                    print("label ID real sample -> ", category,idx_real_label)
                                    break
                    except: 
                        continue
                   
                    
                    # real samples paths, skip samples if contains mixed content (not classifiable)
                    if not(data_model in DF_GROUP_CONTENT['mix']):
                        x_category_real = [os.path.join(path_category_real, name)for name in os.listdir(path_category_real)]
                        y_category_real = [idx_real_label]*len(x_category_real)
                    else:
                        x_category_real = []
                        y_category_real = []
                    
                    # save in global data
                    xs = [*xs, *x_category_real, *x_category_fake]
                    ys = [*ys, *y_category_real, *y_category_fake]
                    
                    if self.train: 
                        n_train += len(x_category_real) + len(x_category_fake)
                    else:
                        n_test  += len(x_category_real) + len(x_category_fake)
                    
            else:                                                               # 2 folder: "0_real", "1_fake"
                path_real = os.path.join(path_set_model, real_folder)
                path_fake = os.path.join(path_set_model, fake_folder)
                # print(path_real,"\n", path_fake)
                
                # get local data
                # fake samples paths
                x_model_fake = [os.path.join(path_fake, name)for name in os.listdir(path_fake)]
                y_model_fake = [idx_fake_label]*len(x_model_fake)
                
                # compute idx real label
                idx_real_label = None
                for k,v in DF_GROUP_CONTENT.items():
                    if data_model in v:
                        if k == "mix": break
                        category = "real_" + k
                        idx_real_label = self.idx2label.index(category)
                        print("label ID real sample -> ", category, idx_real_label)
                        break
                    
                
                
                 # real samples paths, skip samples if contains mixed content (not classifiable)
                if not(data_model in DF_GROUP_CONTENT['mix']):
                    x_model_real = [os.path.join(path_real, name)for name in os.listdir(path_real)]
                    y_model_real = [idx_real_label]*len(x_model_real)
                else:
                    x_model_real = []
                    y_model_real = []
                    
                
                # save in global data
                xs = [*xs, *x_model_real, *x_model_fake]
                ys = [*ys, *y_model_real, *y_model_fake]
                
                if self.train:
                    n_train += len(x_model_real) + len(x_model_fake)
                else:
                    n_test  += len(x_model_real) + len(x_model_fake)    
                
            if self.train: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_train))
            else: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_test))
            
       
        # print the total number of examples
        if self.train:
            print("train samples: {:<10}".format(len(xs)))  
        else:
            print("test samples: {:<10}".format(len(xs)))
            
        self.x = xs
        self.y = ys    # 0-> real, 1 -> fake
    
    def _scanData_allReal(self, real_folder = "0_real", fake_folder = "1_fake"):

        # initialize empty lists
        xs = []  # images path
        ys = []  # binary label for each image
        
        if self.train: print("looking for train data in: {}".format(CDDB_PATH))
        else: print("looking for test data in: {}".format(CDDB_PATH))
        
        data_models = sorted(os.listdir(CDDB_PATH))
        
        # separate ID and OOD
        cateogories = CATEGORIES_SCENARIOS_ID[self.scenario]
            
        ID_groups   = []
        # handle content and group scenario
        if self.scenario == "content" or self.scenario == "group":
            for category in cateogories:
                if self.scenario == "content":
                    ID_groups = [*ID_groups, *DF_GROUP_CONTENT[category]]
                elif self.scenario == "group":
                    ID_groups = [*ID_groups, *DF_GROUP_CLASSES[category]]
                    
        # handle mix scenario  
        else:  
            ID_groups = cateogories

         # get the model needed from the category is contains sub-directory, update In-Distribution group    
        for idx,name in enumerate(ID_groups.copy()):
            if "/" in name:
                ID_groups[idx] = name.split("/")[0]   # take just the model
        
        # split OOD and ID models name
        data_models_ID = []; data_models_OOD = []
        for data_model_name in data_models:
            if data_model_name in ID_groups:
                data_models_ID.append(data_model_name)
            else:
                data_models_OOD.append(data_model_name)
        
        # set name models to scan 
        if self.ood == False:
            data_models = data_models_ID
        else:
            data_models = data_models_OOD 
        
        # shrink the idx2label list
        removed = 0
        for idx, label in enumerate(self.idx2label.copy()):
            if "_".join(label.split("_")[:-1]) in data_models:     # simple check if the name model is present in the idx2label despite "_real" or "_fake" specifics.
                continue
            else:
                self.idx2label.pop(idx - removed)
                removed += 1
        
        # print(self.idx2label)
        
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
            
            
            # take the integer value used to represent the fake and real class for the current model
            idx_fake_label = self.idx2label.index(data_model + "_fake")
            idx_real_label = self.idx2label.index(data_model + "_real")

            # print(idx_fake_label)
            # print(idx_real_label)
            
            if not (real_folder in sub_dir_model and fake_folder in sub_dir_model):   # contains sub-categories
                for category in sub_dir_model:
                    path_category =  os.path.join(path_set_model, category)
                    path_category_real = os.path.join(path_category, real_folder)
                    path_category_fake = os.path.join(path_category, fake_folder)
                    # print(path_category_real, "\n", path_category_fake)
                    
                    # get local data
                    x_category_real = [os.path.join(path_category_real, name)for name in os.listdir(path_category_real)]
                    x_category_fake = [os.path.join(path_category_fake, name)for name in os.listdir(path_category_fake)]
                    y_category_real = [idx_real_label]*len(x_category_real)
                    y_category_fake = [idx_fake_label]*len(x_category_fake)
                    
                    # save in global data
                    xs = [*xs, *x_category_real, *x_category_fake]
                    ys = [*ys, *y_category_real, *y_category_fake]
                    
                    if self.train: 
                        n_train += len(x_category_real) + len(x_category_fake)
                    else:
                        n_test  += len(x_category_real) + len(x_category_fake)
                    
            else:                                                               # 2 folder: "0_real", "1_fake"
                path_real = os.path.join(path_set_model, real_folder)
                path_fake = os.path.join(path_set_model, fake_folder)
                # print(path_real,"\n", path_fake)
                
                # get local data
                x_model_real = [os.path.join(path_real, name)for name in os.listdir(path_real)]
                x_model_fake = [os.path.join(path_fake, name)for name in os.listdir(path_fake)]
                y_model_real = [idx_real_label]*len(x_model_real)
                y_model_fake = [idx_fake_label]*len(x_model_fake)
                
                # save in global data
                xs = [*xs, *x_model_real, *x_model_fake]
                ys = [*ys, *y_model_real, *y_model_fake]
                
                if self.train:
                    n_train += len(x_model_real) + len(x_model_fake)
                else:
                    n_test  += len(x_model_real) + len(x_model_fake)    
                
            if self.train: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_train))
            else: print("found data from {:<20}, train samples -> {:<10}".format(data_model, n_test))
            
       
        # print the total number of examples
        if self.train:
            print("train samples: {:<10}".format(len(xs)))  
        else:
            print("test samples: {:<10}".format(len(xs)))
            
        self.x = xs
        self.y = ys    # 0-> real, 1 -> fake
    
    def _transform(self,x):
        return self.transform_ops(x)
    
    def _one_hot_encoding(self, label_idx):
        encoding = [0] * len(self.idx2label)
        encoding[label_idx] = 1
        return encoding
     
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img_path = self.x[idx]
        # print(img_path)
        img = Image.open(img_path)
        img = self._transform(img)    #dtype = float32
        
        
        # check whether grayscale image, perform pseudocolor inversion 
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)

        # sample the label
        label = self.y[idx]    # if it's necessary the encoding, use self._one_hot_encoding(self.y[idx])
        return img, label

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
    
    # create folder if not exists and download locally
    if not(os.path.exists(CIFAR100_PATH)):
        os.mkdir(CIFAR100_PATH)
        torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=True,  download = True, transform=transform_ops)
        torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=False, download = True, transform=transform_ops)
    
    # load cifar data
    if train:
        cifar100 = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=True, download = False, transform=transform_ops)
    else:
        cifar100 = torchvision.datasets.CIFAR100(root=CIFAR100_PATH, train=False, download = False, transform=transform_ops)
    
    return cifar100

def getMNIST_dataset(train, width_img = 28, height_img = 28):
    """ get MNIST dataset

    Args:
        train (bool): choose between the train or test set. Defaults to True.
        width_img (int, optional): img width for the resize. Defaults to 28.
        height_img (int, optional): img height for the resize. Defaults to 28.
        
    Returns:
        torch.Dataset : Cifar100 dataset object
    """
    transform_ops = transforms.Compose([
        transforms.Resize((width_img, height_img), interpolation= InterpolationMode.BICUBIC, antialias= True),
        transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
    ])
    
    # create folder if not exists and download the dataset
    if not(os.path.exists(MNIST_PATH)):
        os.mkdir(MNIST_PATH)
        torchvision.datasets.MNIST(root=MNIST_PATH, train=True, download = True, transform=transform_ops)
        torchvision.datasets.MNIST(root=MNIST_PATH, train=False, download = True, transform=transform_ops)
    
    # load cifar data
    if train:
        mnist = torchvision.datasets.MNIST(root=MNIST_PATH, train=True, download=False, transform=transform_ops)
    else:
        mnist = torchvision.datasets.MNIST(root=MNIST_PATH, train=False, download=False, transform=transform_ops)
    
    return mnist
    
def getFashionMNIST_dataset(train, width_img = 28, height_img = 28):
    """ get MNIST dataset

    Args:
        train (bool): choose between the train or test set. Defaults to True.
        width_img (int, optional): img width for the resize. Defaults to 28.
        height_img (int, optional): img height for the resize. Defaults to 28.
        
    Returns:
        torch.Dataset : Cifar100 dataset object
    """
    transform_ops = transforms.Compose([
        transforms.Resize((width_img, height_img), interpolation= InterpolationMode.BICUBIC, antialias= True),
        transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image, get tensor [C,H,W]
    ])
    
    # create folder if not exists and download the dataset
    if not(os.path.exists(FMNIST_PATH)):
        os.mkdir(FMNIST_PATH)
        torchvision.datasets.MNIST(root=FMNIST_PATH, train=True,  download = True, transform=transform_ops)
        torchvision.datasets.MNIST(root=FMNIST_PATH, train=False, download = True, transform=transform_ops)
    
    # load cifar data
    if train:
        fmnist = torchvision.datasets.MNIST(root=FMNIST_PATH, train=True, download=False, transform=transform_ops)
    else:
        fmnist = torchvision.datasets.MNIST(root=FMNIST_PATH, train=False, download=False, transform=transform_ops)
    
    return fmnist
    
# synthetic datasets using noise distributions
class GaussianNoise(Dataset):
    """Gaussian Noise Dataset"""

    def __init__(self, size=(3, 224, 224), n_samples=10000, mean=0.5, variance=1.0):
        """ 
            size (int,int,int) -> tuple representing size of the sample to be generated, respecting (n° of channels, height, width)
        """
        self.size = size
        self.n_samples = n_samples
        self.mean = mean
        self.variance = variance
        self.data = np.random.normal(loc=self.mean, scale=self.variance, size=(self.n_samples,) + self.size)
        self.data = np.clip(self.data, 0, 1)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]

class UniformNoise(Dataset):
    """Uniform Noise Dataset"""

    def __init__(self, size=(3, 224, 224), n_samples=10000, low=0, high=1):
        """ 
            size (int,int,int) -> tuple representing size of the sample to be generated, respecting (n° of channels, height, width)
        """
        self.size = size
        self.n_samples = n_samples
        self.low = low
        self.high = high
        self.data = np.random.uniform(low=self.low, high=self.high, size=(self.n_samples,) + self.size)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]

# To generate the dataloader containing both ID and OOD data, with the real and fake labels
class OOD_dataset(Dataset):
    def __init__(self, id_data, ood_data, balancing_mode:str = None, exact_samples:int = None):  # balancing_mode = "max","exact" or None
        """ Dataset for ID and OOD data

        Args:
            id_data (torch.Dataset): The dataset used for the In-Distribution data (label: [1,0] or 0)
            ood_data (torch.Dataset):  The dataset used for the Out-Of-Distribution data (label: [0,1] or 1)
            balancing_mode (str, optional): Possible values are the following: "max","exact" or None. Defaults to None.
                max (str): use the maximum number of samples to perfectly balance ID and OOD data
                exact (str):use an exact number of samples for both ID and OOD data (to be specifed with the "exact_samples" parameter)
                None: No balance is providen
            exact_samples (int, optional): Number of exact samples to specify whether the balancing_mode "exact" is chosen. Defaults to None.
        """
        super(OOD_dataset, self).__init__()
    
        # check on the data parameters 
        assert isinstance(id_data, Dataset)
        assert isinstance(ood_data, Dataset)
        
        self.id_data        = id_data
        self.ood_data       = ood_data
        self.balancing_mode = balancing_mode
        self.exact_samples  = exact_samples
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
            # self.n_IDsamples    = math.floor(self.exact_samples/2)
            # self.n_OODsamples   = math.floor(self.exact_samples/2)
            # if self.exact_samples%2 == 1:
            #     self.n_IDsamples += 1
            # self.n_samples      = self.exact_samples
            
            if (self.exact_samples*2) > min(len(self.id_data), len(self.ood_data)):
                raise ValueError("The exact number of samples should be less that both the number of OOD data and ID data")
            
            self.n_IDsamples    = math.floor(self.exact_samples)
            self.n_OODsamples   = math.floor(self.exact_samples)
            self.n_samples      = self.exact_samples * 2
            
            # compute indices
            self.id_indices  =  random.sample(range(len(self.id_data)), self.n_IDsamples)
            self.ood_indices =  random.sample(range(len(self.ood_data)), self.n_OODsamples)
            
        else:  # unbalanced way of mixing ID and OOD data
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
        
        # label to one-hot-encoding
        y_vector    = [0,0]             # zero vector
        y_vector[y] = 1                 # mark with one the correct position: [1,0] -> ID, [0,1]-> OOD
        y_vector    = T.tensor(y_vector) 
        
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
        data = CDDB_binary_Partial(scenario="mix", ood = True, train = True)
        x,y = data.__getitem__(0)
        print(x)
        showImage(x)
        
    def test_multi_CDDB():
        # tmp = CDDB_binary(train = False)
        # data = CDDB(real_grouping= "single")
        # data = CDDB_binary_Partial(scenario="content")
        data = CDDB_Partial(scenario="mix", real_grouping="categories", ood = True)
        print(data.idx2label)
        # x,y = data.__getitem__(7000)
        # print(y, data.idx2label[y])
        # showImage(x)
    
    def test_getters(name):
        
        # use getters
        if name == "cifar":
            dl_train    = getCIFAR100_dataset(train = True)
            dl_test     = getCIFAR100_dataset(train = False)
        elif name == "mnist":
            dl_train    = getMNIST_dataset(train = True)
            dl_test     = getMNIST_dataset(train = False)
        elif name == "fmnist":
            dl_train    = getFashionMNIST_dataset(train = True)
            dl_test     = getFashionMNIST_dataset(train = False)
        else:
            print("The dataset with name {} is not available".format(name))
            
    
        print(f"train samples number: {len(dl_train)}, test samples number {len(dl_test)}")
        
    test_getters(name="fmnist")
