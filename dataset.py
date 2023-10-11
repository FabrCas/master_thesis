import os
from PIL import Image
import torch as T
T.manual_seed(22)
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from torch.utils.data import Dataset



class CDDB_binary(Dataset):
    def __init__(self, width_img= 224, height_img = 224):
        super(CDDB_binary,self).__init__()
        self.width_img = width_img
        self.height_img = height_img
        self.transform_ops = transforms.Compose([
            transforms.Resize((self.width_img, self.height_img), interpolation= InterpolationMode.BILINEAR, antialias= True),
            transforms.ToTensor(),   #this operation also scales values to be between 0 and 1
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # to have pixel values close between -1 and 1 (imagenet distribution)
        ])

    def _transform(self,x):
        x_transformed = self.transform_ops(x)
    
    def __len__(self):
        # return len()
        return 0
    
    def __getitem__(self, idx):
        return None


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
    pass