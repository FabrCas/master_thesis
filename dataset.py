import os
from PIL import Image
import numpy as np


from utilities import * 

# torch import
import torch as T
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
T.manual_seed(22)


CDDB_PATH = "./data/CDDB"


class CDDB_binary(Dataset):
    def __init__(self, width_img= 224, height_img = 224, train = True):
        super(CDDB_binary,self).__init__()
        self.train = train                      # boolean flag to select train or test data
        self.width_img = width_img
        self.height_img = height_img
        self.transform_ops = transforms.Compose([
            transforms.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
            transforms.ToTensor(),   # this operation also scales values to be between 0 and 1, expected [H, W, C] format numpy array or PIL image
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
        ])
        
        # load the data paths and labels
        self.x_train = None; self.y_train = None; self.x_test = None; self.y_test = None
        self._scanData()
        
        # load the correct data 
        if self.train:
            self.x = self.x_train
            self.y = self.y_train   # 0-> real, 1 -> fake
        else:
            self.x = self.x_test
            self.y = self.y_test    # 0-> real, 1 -> fake
        

    def _scanData(self, real_folder = "0_real", fake_folder = "1_fake"):
        
        # initialize empty lists
        xs_train    = [] # images path (train)
        ys_train    = []  # binary label for each image (train)
        xs_test     = [] # images path (test)
        ys_test     = []  # binary label for each image (test)
        
        
        print("looking for data in: {}".format(CDDB_PATH))
        
        data_models = sorted(os.listdir(CDDB_PATH))
        for data_model in data_models:                                      # loop over folders of each model
            path_train_model = os.path.join(CDDB_PATH,data_model,"train")    # around 3/4 of all the data
            path_val_model = os.path.join(CDDB_PATH,data_model,"val")        # around 1/4 of all the data
            
            #                               extract for the train set
            sub_dir_model = sorted(os.listdir(path_train_model))
            # print(sub_dir_model)
            
            # count how many samples you collect
            n_train     = 0
            n_test      = 0
            
            if not (real_folder in sub_dir_model and fake_folder in sub_dir_model):   # contains sub-categories
                for category in sub_dir_model:
                    path_category =  os.path.join(path_train_model, category)
                    path_category_real = os.path.join(path_category, real_folder)
                    path_category_fake = os.path.join(path_category, fake_folder)
                    # print(path_category_real, "\n", path_category_fake)
                    
                    # get local data
                    x_category_real = [os.path.join(path_category_real, name)for name in os.listdir(path_category_real)]
                    x_category_fake = [os.path.join(path_category_fake, name)for name in os.listdir(path_category_fake)]
                    y_category_real = [0]*len(x_category_real)
                    y_category_fake = [1]*len(x_category_fake)
                    
                    # save in global data
                    xs_train = [*xs_train, *x_category_real, *x_category_fake]
                    ys_train = [*ys_train, *y_category_real, *y_category_fake]
                    
                    n_train     = len(x_category_real) + len(x_category_fake)
                    
                    
                    
            else:                                                               # 2 folder: "0_real", "1_fake"
                path_real = os.path.join(path_train_model, real_folder)
                path_fake = os.path.join(path_train_model, fake_folder)
                # print(path_real,"\n", path_fake)
                
                # get local data
                x_model_real = [os.path.join(path_real, name)for name in os.listdir(path_real)]
                x_model_fake = [os.path.join(path_fake, name)for name in os.listdir(path_fake)]
                y_model_real = [0]*len(x_model_real)
                y_model_fake = [1]*len(x_model_fake)
                
                # save in global data
                xs_train = [*xs_train, *x_model_real, *x_model_fake]
                ys_train = [*ys_train, *y_model_real, *y_model_fake]
                
                n_train     = len(x_model_real) + len(x_model_fake)
                
               
            #                               extract for the test set
            sub_dir_model = sorted(os.listdir(path_val_model))            
            # print(sub_dir_model)
            if not (real_folder in sub_dir_model and fake_folder in sub_dir_model):   # contains sub-categories
                for category in sub_dir_model:
                    path_category =  os.path.join(path_val_model, category)
                    path_category_real = os.path.join(path_category, real_folder)
                    path_category_fake = os.path.join(path_category, fake_folder)
                    # print(path_category_real, "\n", path_category_fake)
                    
                    # get local data
                    x_category_real = [os.path.join(path_category_real, name)for name in os.listdir(path_category_real)]
                    x_category_fake = [os.path.join(path_category_fake, name)for name in os.listdir(path_category_fake)]
                    y_category_real = [0]*len(x_category_real)
                    y_category_fake = [1]*len(x_category_fake)
                    
                    # save in global data
                    xs_test = [*xs_test, *x_category_real, *x_category_fake]
                    ys_test = [*ys_test, *y_category_real, *y_category_fake]
                    
                    n_test     = len(x_category_real) + len(x_category_fake)
                    
                    
            else:                                                               # 2 folder: "0_real", "1_fake"
                path_real = os.path.join(path_val_model, real_folder)
                path_fake = os.path.join(path_val_model, fake_folder)
                # print(path_real,"\n", path_fake)
                
                # get local data
                x_model_real = [os.path.join(path_real, name)for name in os.listdir(path_real)]
                x_model_fake = [os.path.join(path_fake, name)for name in os.listdir(path_fake)]
                y_model_real = [0]*len(x_model_real)
                y_model_fake = [1]*len(x_model_fake)
                
                # save in global data
                xs_test = [*xs_test, *x_model_real, *x_model_fake]
                ys_test = [*ys_test, *y_model_real, *y_model_fake]
                
                n_test     = len(x_model_real) + len(x_model_fake)

            print("found data from {:<20}, train -> {:<10}, test -> {:<10}".format(data_model, n_train, n_test))
            
        print("train samples: {:<10}".format(len(xs_train)))  
        print("test samples: {:<10}".format(len(xs_test)))
    
        self.x_train    = xs_train
        self.y_train    = ys_train
        self.x_test     = xs_test
        self.y_test     = ys_test
        
    def _transform(self,x):
        return self.transform_ops(x)

    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img_path = self.x[idx]
        # print(img_path)
        img = Image.open(img_path)
        img = self._transform(img)
        label = self.y[idx]
        return img, label



#TODO
class CDDB():
    def __init__(self, width_img= 224, height_img = 224):
        super(CDDB,self).__init__()
        
        self.width_img = width_img
        self.height_img = height_img
        
        self.transform_ops = transforms.Compose([
            transforms.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
        ])

    def _transform(self,x):
        x_transformed = self.transform_ops(x)
    
    def __len__(self):
        # return len()
        return 0
    
    def __getitem__(self, idx):
        return None


        
# [test section]
if __name__ == "__main__":
    dataset = CDDB_binary(train= True)
    
    # test Dataset item get
    x,y = dataset.__getitem__(0)
    print(y)
    print(x.shape)
    showImage(x, name = "CDDB sample")
    from torch.utils.data import DataLoader
    
    # test Dataloader from Dataset
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle= True, drop_last= True, pin_memory= True)
    for x,y in dataloader:
        print(x.shape)
        print(y)
        for i in range(x.shape[0]):
            img = x[i]
            showImage(img)

        break
        
    
    
    
    