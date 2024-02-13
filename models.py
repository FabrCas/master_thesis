import time

import  torch                           as T
import  torch.nn.functional             as F
import  numpy                           as np
import  math
import  torch.nn                        as nn
from    PIL                             import Image

from    torchsummary                    import summary
from    torchvision                     import models
from    torchvision                     import transforms
from    torchvision.models              import ResNet50_Weights, ViT_B_16_Weights
from    utilities                       import print_dict, print_list, expand_encoding, convTranspose2d_shapes, get_inputConfig, \
                                            showImage, trans_input_base, alpha_blend_pytorch, include_attention
from    einops.layers.torch             import Rearrange
from    einops                          import repeat, rearrange
# import  cv2
import  timm

T.manual_seed(seed=22)

# input settigs:
config = get_inputConfig()

INPUT_WIDTH     = config['width']
INPUT_HEIGHT    = config['height']
INPUT_CHANNELS  = config['channels']

# 1st models superclass for deepfake detection
class Project_DFD_model(nn.Module):
    
    def __init__(self, c,h,w, n_classes):
        super(Project_DFD_model,self).__init__()
        print("Initializing {} ...".format(self.__class__.__name__))
        
        # initialize the model to None        
        self.input_dim  = (INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
        
        # features, classes and spatial dimensions
        self.n_classes      = n_classes
        self.n_channels     = c
        self.width          = h
        self.height         = w
        self.input_shape    = (self.n_channels, self.height, self.width)   # input_shape doesn't consider the batch

    
    def getSummary(self, input_shape = None, verbose = True):  #shape: color,width,height
        """
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this type -> color,width,height
        """
        
        if input_shape is None:
            input_shape = (self.n_channels, self.height, self.width)
            
        try:
            model_stats = summary(self, input_shape, verbose = int(verbose))
            return str(model_stats)
        except Exception as e:
            # print(e)
            summ = ""
            n_params = 0
            for k,v in self.getLayers().items():
                summ += "{:<50} -> {:<50}".format(k,str(tuple(v.shape))) + "\n"
                n_params += T.numel(v)
            summ += "Total number of parameters: {}\n".format(n_params)
            if verbose: print(summ)
            return summ
    
    def getLayers(self):
        return dict(self.named_parameters())
    
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
    
    def getDevice(self):
        return next(self.parameters()).device
    
    def isCuda(self):
        return next(self.parameters()).is_cuda
    
    def to_device(self, device):   # alias for to(device) function of nn.Module
        self.to(device)
        
    def _init_weights_kaimingNormal(self):
        # Initialize the weights  using He initialization (good for conv net)
        print("Weights initialization using kaiming Normal")
               
        for param in self.parameters():
            if len(param.shape) > 1:
                T.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    
    def _init_weights_normal(self):
        print(f"Weights initialization using Gaussian distribution")
        # Initialize the weights with Gaussian distribution
        for param in self.parameters():
            if len(param.shape) > 1:
                T.nn.init.normal_(param, mean=0, std=0.01) 
    
    def _init_weights_kaimingNormal_module(self, model = None):
        # Initialize the weights  using He initialization
        print("Weights initialization using kaiming Normal")
        
        if model is None: model = self
        
        for param in model.parameters():
            if len(param.shape) > 1:
                T.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    
    def _init_weights_normal_module(self, model = None):
        # Initialize the weights with Gaussian distribution
        print(f"Weights initialization using Gaussian distribution")
        
        if model is None: model = self
        for param in model.parameters():
            if len(param.shape) > 1:
                T.nn.init.normal_(param, mean=0, std=0.01) 
    
    def getAttributes(self):
        att = self.__dict__
        
        def valid(pair):
            # unzip pair k,v
            _, x = pair 
            type_x = type(x)
            condition =  (type_x is int) or (type_x is str) or (type_x is tuple)
            return condition
        
        filterd_att = dict(filter(valid, att.items()))
        return filterd_att
    
    def forward(self):
        raise NotImplementedError

#_________________________________________ ResNet__________________________________________________

#                                       ResNet 50 ImageNet
class ResNet_ImageNet(Project_DFD_model):   # not nn.Module subclass, but still implement forward method calling the one of the model
    """ 
    This is a wrap class for pretraiend Resnet use the getModel function to get the nn.module implementation.
    The model expects color images in RGB standard, of size 244x244
    """
    
    
    def __init__(self, n_classes = 10):
        super(ResNet_ImageNet,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.weight_name =  ResNet50_Weights.IMAGENET1K_V2  # weights with accuracy 80.858% on ImageNet 
        self.model = self._create_net()
        
    def _create_net(self):
        model = models.resnet50(weights= self.weight_name)
        
        # edit first layer to accept grayscale images
        if self.n_channels == 1:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # edit fully connected layer for the output
        model.fc = nn.Linear(model.fc.in_features, self.n_classes)
        return model
        
    def getModel(self):
        return self.model
    
    def getSummary(self, input_shape = None, verbose = True):  #shape: color,width,height
        """
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this type -> color,width,height
        """
        
        if input_shape is None:
            input_shape = (self.n_channels, self.height, self.width)
        
        model_stats = summary(self.model, input_shape, verbose = int(verbose))
        return str(model_stats)
     
    def to(self, device):
        self.model.to(device)
    
    def getLayers(self):
        return dict(self.model.named_parameters())
    
    def freeze(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
    
    def isCuda(self):
        return next(self.model.parameters()).is_cuda
         
    def forward(self, x):
        x = self.model(x)
        return x

#                                       custom ResNet with encodign and image reconstruction (OOD detection related)
class ResNet_EDS(Project_DFD_model): 
    """ ResNet multi head module with Encoder, Decoder and scorer (Classifier) """
    def __init__(self, n_classes = 10, use_upsample = False):               # expect image of shape 224x224
        super(ResNet_EDS,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.use_upsample = use_upsample
        self.weight_name =  ResNet50_Weights.IMAGENET1K_V2  # weights with accuracy 80.858% on ImageNet 
        self._createModel()
        
    def _createModel(self):
                
        # load pre-trained ResNet50
        self.encoder_module = models.resnet50(weights= self.weight_name)
        # replace RelU with GELU function
        self._replaceReLU(self.encoder_module)
        
        # edit first layer to accept grayscale images if its the case
        if self.n_channels == 1:
            self.encoder_module.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # define the bottleneck dimension
        bottleneck_size = self.encoder_module.fc.in_features
        
        # remove last fully connected layer for classification
        self.encoder_module.fc = nn.Identity()
        # self.encoder_module = model
        
        # Define the classification head
        self.scorer_module = nn.Linear(bottleneck_size, self.n_classes)
        
        print("bottleneck size is: {}".format(bottleneck_size))
        
        self.decoder_out_fn= nn.Sigmoid()
    
        # Define the Decoder (self.use_upsample defines in which way increase spatial dimension from 1x1 to 7x7)
        if self.use_upsample:   
            self.decoder_module = nn.Sequential(
                expand_encoding(),   # to pass from [-1,2048] to [-1,2048,1,1]
                
                nn.Upsample(scale_factor=7),
                nn.ConvTranspose2d(bottleneck_size, 128, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=128),
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=64),
                
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=32),
                
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=16),
                
                nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=3),
                
                self.decoder_out_fn             # to have pixels in the range of 0,1  (or  nn.Tanh)
            )
        else:
            self.decoder_module = nn.Sequential(
                expand_encoding(),   # to pass from [-1,2048] to [-1,2048,1,1]
                
                nn.ConvTranspose2d(bottleneck_size, 128, kernel_size=5, stride=3, padding=0, output_padding= 2),
                nn.GELU(),
                self.conv_block_decoder(n_filter=128),
                
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=64),
                
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=32),
                
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=16),
                
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=8),
                
                nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                self.conv_block_decoder(n_filter=3),
                
                self.decoder_out_fn             # to have pixels in the range of 0,1  (or  nn.Tanh)
            )
        
        # initialize scorer and decoder  (not initialize the encoder since is pre-trained!)
        self.init_weights_normal(self.scorer_module)
        self.init_weights_kaimingNormal(self.decoder_module)
        
    # aux functions to create/initialize the model
    
    def conv_block_decoder(self, n_filter):
        # U-net inspired structures
        
        conv_block = nn.Sequential(
                # Taking first input and implementing the first conv block
                nn.Conv2d(n_filter, n_filter,kernel_size = 3, padding = "same"),
                nn.BatchNorm2d(n_filter),
                nn.GELU(),

                # Taking first input and implementing the second conv block
                nn.Conv2d(n_filter, n_filter,kernel_size = 3, padding = "same"),
                nn.BatchNorm2d(n_filter),
                nn.GELU(),
        )
        return conv_block
    
    def _replaceReLU(self, model, verbose = False):
        """ function used to replace the ReLU acitvation function with another one, default is the GELU"""
        
        full_name = ""
        for name, m in model.named_children():
            full_name = f"{full_name}.{name}"

            if isinstance(m, nn.ReLU):
                setattr(model, name, nn.GELU())
                if verbose: print(f"replaced {full_name}: {nn.ReLU}->{nn.GELU}")
                
            elif len(list(m.children())) > 0:
                self._replaceReLU(m)
    
    def _expand_encoding(x):
        return x.view(-1, 2048, 1, 1)   # 2048 is the encoding length
    
    def init_weights_normal(self, model):
        print(f"Weights initialization using Gaussian distribution")
        # Initialize the weights with Gaussian distribution
        for param in model.parameters():
            if len(param.shape) > 1:
                T.nn.init.normal_(param, mean=0, std=0.01) 
                
    def init_weights_kaimingNormal(self, model):
        # Initialize the weights  using He initialization
        print("Weights initialization using kaiming Normal")
        for param in model.parameters():
            if len(param.shape) > 1:
                T.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    
    def reparameterize(self, z):
        z = z.view(z.size(0), -1)
        mu, log_sigma = z[:, :self.z_dim], z[:, self.z_dim:]
        std = T.exp(log_sigma)
        eps = T.randn_like(std)
        return mu + eps * std
    
    # getters methods 
    
    def getEncoder_module(self):
        return self.encoder_module
    
    def getDecoder_module(self):
        return self.decoder_module

    def getScorer_module(self):
        return self.scorer_module
    
    def getSummaryEncoder(self, input_shape = None):  #shape: color,width,height
        """
            summary for encoder
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this type -> color,width,height
        """
        
        if input_shape is None:
            input_shape = (self.n_channels, self.height, self.width)
            
        model_stats = summary(self.encoder_module, input_shape, verbose =0)
        return str(model_stats)
    
    def getSummaryScorerPipeline(self, input_shape = None):
        """
            summary for encoder + scoder modules
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this shape -> color,width,height
        """
        if input_shape is None:
            input_shape = (self.n_channels, self.height, self.width)
        
        model_stats = summary(nn.Sequential(self.encoder_module, self.scorer_module), input_shape, verbose = 0)
        return str(model_stats)
    
    def getSummaryDecoderPipeline(self, input_shape = None):
        """
            summary for encoder + decoder modules
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this shape -> color,width,height
        """
        if input_shape is None:
            input_shape = (self.n_channels, self.height, self.width)
            
        model_stats = summary(nn.Sequential(self.encoder_module, self.decoder_module), input_shape, verbose = 0)
        return str(model_stats)
  
    def getSummary(self, input_shape = (3,244,244), verbose = True):
        """
            summary for the whole model
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this type -> color,width,height
        """
        model_stats_encoder     = self.getSummaryEncoder()
        model_stats_scorer      = self.getSummaryScorerPipeline()
        model_stats_decoder     = self.getSummaryDecoderPipeline() 
        
        stats = "\n\n{:^90}\n\n\n".format("Encoder:") + model_stats_encoder + "\n\n\n{:^90}\n\n\n".format("Scorer Pipeline:") + \
                model_stats_scorer + "\n\n\n{:^90}\n\n\n".format("Decoder Pipeline:") + model_stats_decoder
        
        if verbose: print(stats)
        
        return stats

    def getLayers(self, name_module = "encoder"):
        """
            return the layers of the selected module
            
            Args: 
                name_module(str) choose between: "encoder", "scorer" and "decoder"
        """
        if name_module == "encoder":
            return dict(self.encoder_module.named_parameters())
        elif name_module == "scorer":
            return dict(self.scorer_module.named_parameters())
        elif name_module == "decoder":
            return dict(self.decoder_module.named_parameters())  
        else:
            return dict(self.named_parameters()) 
    
    def getDevice(self, name_module):
        """
            return the device assigned for the selected module
            
            Args: 
                name_module(str) choose between: "encoder", "scorer" and "decoder"
        """
        if name_module == "encoder":
            return next(self.encoder_module.parameters()).device
        elif name_module == "scorer":
            return next(self.scorer_module.parameters()).device
        elif name_module == "decoder":
            return next(self.decoder_module.parameters()).device
      
    # set and forward methods
    
    def to(self, device):
        """ move the whole model to "device" (str) """
        
        self.encoder_module.to(device)
        self.scorer_module.to(device)
        self.decoder_module.to(device)
        
        # self.scorer_pipe.to(device)
        # self.decoder_pipe.to(device)
     
    def freeze(self, name_module):
        """
            freeze the layers of the selected module
            
            Args: 
                name_module(str) choose between: "encoder", "scorer" and "decoder" or "all"
        """
        if name_module == "encoder":
            for name, param in self.encoder_module.named_parameters():
                param.requires_grad = False
        elif name_module == "scorer":
            for name, param in self.scorer_module.named_parameters():
                param.requires_grad = False
        elif name_module == "decoder":
            for name, param in self.decoder_module.named_parameters():
                param.requires_grad = False
        elif name_module == "all":
            self.freeze("encoder")
            self.freeze("decoder")
            self.freeze("scorer")
    
    def forward(self, x):
        """
            Forward of the module
            
            Args: 
                x (T.tensor) batch or image tensor
            
            Returns:
                3xT.tensor: features from encoder, logits from scorer and reconstruction from decoder
        """
        # encoder forward
        features = self.encoder_module(x)
        
        # classification forward
        logits = self.scorer_module(features)
        
        # decoder forward
        reconstruction = self.decoder_module(features)
        
        return features, logits, reconstruction

    
# _____________________________________ U net ______________________________________________________

# Default U-net based model setting:
UNET_EXP_FMS    = 4    # U-net power of 2 exponent for feature maps (starting)

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = Conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
        # return p

class Decoder_block(nn.Module):
    def __init__(self, in_c, out_c, out_pad = 0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0, output_padding = out_pad)
        self.conv = Conv_block(out_c+out_c, out_c)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = T.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class Unet4(Project_DFD_model):
    """
        U-net 4, 4 encoders and 4 decoders
    """
    
    def __init__(self):
        super(Unet4,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = None)
        
        print("Initializing {} ...".format(self.__class__.__name__))
        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.n_levels = 4
        
        # create net and initialize
        self._createNet()
        self._init_weights_kaimingNormal()
        

        
    def _createNet(self):
        # encoder
        self.e1 = Encoder_block(self.n_channels, self.feature_maps(0))
        self.e2 = Encoder_block(self.feature_maps(0) , self.feature_maps(1))
        self.e3 = Encoder_block(self.feature_maps(1) , self.feature_maps(2))
        self.e4 = Encoder_block(self.feature_maps(2) , self.feature_maps(3))
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(3) , self.feature_maps(4))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
    
        # decoder 
        self.d1 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d2 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d3 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d4 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()
            
        # self.model = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x):
        """
            Returns: reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
    
        # bottlenech (encoding)
        bottleneck = self.b(p4)
        enc = self.flatten(bottleneck)

        # decoder 
        d1 = self.d1(bottleneck, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # reconstuction
        rec = self.decoder_out_fn(self.out(d4))  # check sigmoid vs tanh
    
        return rec, enc

class Unet5(Project_DFD_model):
    """
        U-net 5, 5 encoders and 5 decoders
    """
    
    def __init__(self):
        super(Unet5,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = None)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.n_levels = 5
        
        # create net and initialize
        self._createNet()
        self._init_weights_kaimingNormal()

        
    def _createNet(self):
        
        # encoder
        self.e1 = Encoder_block(self.n_channels, self.feature_maps(0))
        self.e2 = Encoder_block(self.feature_maps(0) , self.feature_maps(1))
        self.e3 = Encoder_block(self.feature_maps(1) , self.feature_maps(2))
        self.e4 = Encoder_block(self.feature_maps(2) , self.feature_maps(3))
        self.e5 = Encoder_block(self.feature_maps(3) , self.feature_maps(4))
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(4) , self.feature_maps(5))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
    
        # decoder 
        
        # define conditions for padding 
        c = INPUT_WIDTH/(math.pow(2,self.n_levels))
        
        if c%1!=0:
            self.d1 = Decoder_block(self.feature_maps(5) , self.feature_maps(4), out_pad=1) # 112x112 addition
        else:
            self.d1 = Decoder_block(self.feature_maps(5) , self.feature_maps(4))
        self.d2 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d3 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d4 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d5 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()
            
        # self.model = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x):
        """
            Returns: reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        
        # bottlenech (encoding)
        bottleneck = self.b(p5)
        enc = self.flatten(bottleneck)
        
        print(p1.shape)
        print(p2.shape)
        print(p3.shape)
        print(p4.shape)
        print(p5.shape)
        print(bottleneck.shape)
        
        
        
        # decoder 
        d1 = self.d1(bottleneck, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)
    
        # reconstuction
        rec = self.decoder_out_fn(self.out(d5))  # check sigmoid vs tanh
    
        return rec, enc

class Unet6(Project_DFD_model):
    """
        U-net 6, 6 encoders and 6 decoders
    """
    
    def __init__(self):
        super(Unet6,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = None)
        print("Initializing {} ...".format(self.__class__.__name__))
        
        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.n_levels = 6
        
        # create net and initialize
        self._createNet()
        self._init_weights_kaimingNormal()
        

        
    def _createNet(self):
        
        # encoder
        self.e1 = Encoder_block(self.n_channels, self.feature_maps(0))
        self.e2 = Encoder_block(self.feature_maps(0) , self.feature_maps(1))
        self.e3 = Encoder_block(self.feature_maps(1) , self.feature_maps(2))
        self.e4 = Encoder_block(self.feature_maps(2) , self.feature_maps(3))
        self.e5 = Encoder_block(self.feature_maps(3) , self.feature_maps(4))
        self.e6 = Encoder_block(self.feature_maps(4) , self.feature_maps(5))
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(5) , self.feature_maps(6))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
    
        # decoder 
        
        # define conditions for padding 
        c1 = INPUT_WIDTH/(math.pow(2,self.n_levels))
        c2 = INPUT_WIDTH/(math.pow(2,self.n_levels-1))
        
        if c1%1!=0:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5), out_pad=1) # for odd spatial dimensions
        else:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5)) # for odd spatial dimensions
        if c2%1!=0:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4), out_pad=1) # 112x112 addition
        else:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4)) # 112x112 addition
        self.d3 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d4 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d5 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d6 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()
            
        # self.model = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x):
        """
            Returns: reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        s6, p6 = self.e6(p5)
        
        # bottlenech (encoding)
        bottleneck = self.b(p6)
        enc = self.flatten(bottleneck)

        # decoder 
        d1 = self.d1(bottleneck, s6)
        d2 = self.d2(d1, s5)
        d3 = self.d3(d2, s4)
        d4 = self.d4(d3, s3)
        d5 = self.d5(d4, s2)
        d6 = self.d6(d5, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d6))  # check sigmoid vs tanh
    
        return rec, enc

    
#_____________________________________ OOD custom models: Unet based ___________________

#                                       custom Unet

# -- residual block definition
class Encoder_block_residual(nn.Module):
    def __init__(self, in_c, out_c, after_conv = False, kernel_size = 1):
        super().__init__()
        self.after_conv = after_conv
        
        # Downsample layer for identity shortcut
        if self.after_conv:          # after conv, no downsampling just adjustment feature maps
            self.ds_layer = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size = kernel_size, stride=1),
                nn.BatchNorm2d(out_c)
            )

        else:                           # after pooling
            self.ds_layer = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size= kernel_size, stride=2),
                nn.BatchNorm2d(out_c)
            )
        
        self.conv = Conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, inputs):
        inputs_init  = inputs.clone()
        
        x = self.conv(inputs)
        
        # identity shortcuts after conv block
        if self.after_conv:
            inputs_converted =  self.ds_layer(inputs_init)
            x += inputs_converted
        
        p = self.pool(x)
        
        # print("pooling -> ", p.shape)
        
        # identity shortcuts after pooling
        if not(self.after_conv):
            inputs_downsampled =  self.ds_layer(inputs_init)
            # print("x_downsampled ->", inputs_downsampled.shape)
            
            p += inputs_downsampled
        
        return x, p

# -- large blocks definition
class LargeConv_block(nn.Module):
    # TODO test other version of large conv since results fail to increase with this

    """ larger version of the Conv_block class"""
    def __init__(self, in_c, out_c):
        super().__init__()
        
        # self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(in_c)
        # self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_c)
        # self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
        # self.bn3 = nn.BatchNorm2d(out_c)
        
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x 

class LargeEncoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_1 = LargeConv_block(in_c, in_c)
        self.conv_2 = LargeConv_block(in_c, out_c)
        self.conv_3 = LargeConv_block(out_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        p = self.pool(x)
        return x, p

class LargeEncoder_block_residual(nn.Module):
    def __init__(self, in_c, out_c, after_conv = False, kernel_size = 1):
        super().__init__()
        self.after_conv = after_conv
        
        # Downsample layer for identity shortcut
        if self.after_conv:          # after conv, no downsampling just adjustment feature maps
            self.ds_layer = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size = kernel_size, stride=1),
                nn.BatchNorm2d(out_c)
            )

        else:                           # after pooling
            self.ds_layer = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size= kernel_size, stride=2),
                nn.BatchNorm2d(out_c)
            )
        
        self.conv = LargeConv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, inputs):
        inputs_init  = inputs.clone()
        
        x = self.conv(inputs)
        
        # identity shortcuts after conv block
        if self.after_conv:
            inputs_converted =  self.ds_layer(inputs_init)
            x += inputs_converted
        
        p = self.pool(x)
        
        # print("pooling -> ", p.shape)
        
        # identity shortcuts after pooling
        if not(self.after_conv):
            inputs_downsampled =  self.ds_layer(inputs_init)
            # print("x_downsampled ->", inputs_downsampled.shape)
            
            p += inputs_downsampled
        
        return x, p

# -- Unet models 
class Unet4_Scorer(Project_DFD_model):
    """
        U-net 4 + Scorer, 4 encoders and 4 decoders
    """

    def __init__(self, n_classes= 10,  large_encoding = True):
        super(Unet4_Scorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes,)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        # self.bottleneck_size = int(self.feature_maps(4)*(self.width/16)*(self.width/16))
        self.bottleneck_size = int(self.feature_maps(4)*math.floor(self.width/16)**2) 
        self.n_levels = 4
        self.large_encoding = large_encoding
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal_module()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
    
    def _createNet(self):
        # encoder
        self.e1 = Encoder_block(self.n_channels, self.feature_maps(0))
        self.e2 = Encoder_block(self.feature_maps(0) , self.feature_maps(1))
        self.e3 = Encoder_block(self.feature_maps(1) , self.feature_maps(2))
        self.e4 = Encoder_block(self.feature_maps(2) , self.feature_maps(3))
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(3) , self.feature_maps(4))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
        
        # self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/16))
        # self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/16))
        # self.fc2 = nn.Linear(int(self.bottleneck_size/16), int(self.bottleneck_size/64))
        # self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        # self.fc3 = nn.Linear(int(self.bottleneck_size/64), self.n_classes)
        
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        
        
        # decoder 
        self.d1 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d2 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d3 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d4 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()
            
        # self.model = nn.Sequential(self.encoder, self.decoder)


    def forward(self, x, verbose = False):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
    
        # bottleneck (encoding)
        bottleneck = self.b(p4)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)

        
        # classification
        
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)
        
        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")
        
        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1
        

        # decoder 
        d1 = self.d1(bottleneck, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d4))  # check sigmoid vs tanh
        return logits, rec, encoding

class Unet5_Scorer(Project_DFD_model):
    """
        U-net 5 + Scorer, 5 encoders and 5 decoders
    """

    def __init__(self, n_classes= 10, large_encoding = True):
        super(Unet5_Scorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.bottleneck_size = int(self.feature_maps(5)*math.floor(self.width/32)**2)  
        self.n_levels = 5
        self.large_encoding = large_encoding       
        
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
    
    def _createNet(self):
        
        # encoder
        self.e1 = Encoder_block(self.n_channels, self.feature_maps(0))
        self.e2 = Encoder_block(self.feature_maps(0) , self.feature_maps(1))
        self.e3 = Encoder_block(self.feature_maps(1) , self.feature_maps(2))
        self.e4 = Encoder_block(self.feature_maps(2) , self.feature_maps(3))
        self.e5 = Encoder_block(self.feature_maps(3) , self.feature_maps(4))
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(4) , self.feature_maps(5))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
        
        # self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/16))
        # self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/16))
        # self.fc2 = nn.Linear(int(self.bottleneck_size/16), int(self.bottleneck_size/64))
        # self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        # self.fc3 = nn.Linear(int(self.bottleneck_size/64), self.n_classes)
        
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        
        # decoder 
        
        # define conditions for padding 
        c = INPUT_WIDTH/(math.pow(2,self.n_levels))
        
        if c%1!=0:
            self.d1 = Decoder_block(self.feature_maps(5) , self.feature_maps(4), out_pad=1) # 112x112 addition
        else:
            self.d1 = Decoder_block(self.feature_maps(5) , self.feature_maps(4))
        self.d2 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d3 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d4 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d5 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()
            
        # self.model = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x, verbose = False):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
    
        # bottleneck (encoding)
        bottleneck  = self.b(p5)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)
        
        # classification
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)
        
        
        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")
            
        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1
        
        
        # decoder 
        d1 = self.d1(bottleneck, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d5))  # check sigmoid vs tanh
        return logits, rec, encoding

class Unet6_Scorer(Project_DFD_model):
    """
        U-net 6 + Scorer, 6 encoders and 6 decoders.
        This version include an additional layer for the scorer
    """

    def __init__(self, n_classes= 10, large_encoding = True):
        super(Unet6_Scorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.n_levels = 6
        self.large_encoding = large_encoding
        
        # self.bottleneck_size = int(self.feature_maps(6)*3**2)  # 3 comes from the computation on spatial dimensionality, kernel applied on odd tensor (if image different from 244 check again this value)
        self.bottleneck_size = int(self.feature_maps(6)*math.floor(self.width/64)**2)  # if 224x224 the width is divided by 32
        
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
    
    def _createNet(self):
        
        # encoder
        self.e1 = Encoder_block(self.n_channels, self.feature_maps(0))
        self.e2 = Encoder_block(self.feature_maps(0) , self.feature_maps(1))
        self.e3 = Encoder_block(self.feature_maps(1) , self.feature_maps(2))
        self.e4 = Encoder_block(self.feature_maps(2) , self.feature_maps(3))
        self.e5 = Encoder_block(self.feature_maps(3) , self.feature_maps(4))
        self.e6 = Encoder_block(self.feature_maps(4) , self.feature_maps(5))
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(5) , self.feature_maps(6))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
        
        # series of FC layers
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        # decoder 
        
        # define conditions for padding 
        c1 = INPUT_WIDTH/(math.pow(2,self.n_levels))
        c2 = INPUT_WIDTH/(math.pow(2,self.n_levels-1))
        
        if c1%1!=0:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5), out_pad=1) # for odd spatial dimensions
        else:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5)) # for odd spatial dimensions
        if c2%1!=0:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4), out_pad=1) # 112x112 addition
        else:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4)) # 112x112 addition
        self.d3 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d4 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d5 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d6 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()


    def forward(self, x, verbose = False):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        s6, p6 = self.e6(p5)
    
        # bottleneck (encoding)
        bottleneck  = self.b(p6)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)
        # classification
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)

        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")
        
        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1
            
        d1 = self.d1(bottleneck, s6)
        d2 = self.d2(d1, s5)
        d3 = self.d3(d2, s4)
        d4 = self.d4(d3, s3)
        d5 = self.d5(d4, s2)
        d6 = self.d6(d5, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d6))  # check sigmoid vs tanh
        return logits, rec, encoding

class Unet6L_Scorer(Project_DFD_model):
    """
        U-net 6 + Scorer, 6 encoders and 6 decoders.
        This version include an additional layer for the scorer and LargeConv_block instead of Conv_blocks
    """

    def __init__(self, n_classes= 10, large_encoding = True):
        super(Unet6L_Scorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.bottleneck_size = int(self.feature_maps(6)*math.floor(self.width/64)**2)
        self.n_levels = 6
        self.large_encoding = large_encoding
        
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
    
    def _createNet(self):
        
        # encoder
        self.e1 = LargeEncoder_block(self.n_channels, self.feature_maps(0))
        self.e2 = LargeEncoder_block(self.feature_maps(0) , self.feature_maps(1))
        self.e3 = LargeEncoder_block(self.feature_maps(1) , self.feature_maps(2))
        self.e4 = LargeEncoder_block(self.feature_maps(2) , self.feature_maps(3))
        self.e5 = LargeEncoder_block(self.feature_maps(3) , self.feature_maps(4))
        self.e6 = LargeEncoder_block(self.feature_maps(4) , self.feature_maps(5))
        
        # bottlenech (encoding)
        self.b = LargeConv_block(self.feature_maps(5) , self.feature_maps(6))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
        
        # series of FC layers
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        # decoder 
        
        # define conditions for padding 
        c1 = INPUT_WIDTH/(math.pow(2,self.n_levels))
        c2 = INPUT_WIDTH/(math.pow(2,self.n_levels-1))
        
        if c1%1!=0:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5), out_pad=1) # for odd spatial dimensions
        else:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5)) # for odd spatial dimensions
        if c2%1!=0:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4), out_pad=1) # 112x112 addition
        else:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4)) # 112x112 addition
        self.d3 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d4 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d5 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d6 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()

    
    def forward(self, x, verbose = False):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        s6, p6 = self.e6(p5)
    
        # bottleneck (encoding)
        bottleneck  = self.b(p6)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)
        
        # print(bottleneck.shape)
        # print(enc.shape)
        # print(self.bottleneck_size)
        
        # classification
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)
        
        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")

        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1
            
        d1 = self.d1(bottleneck, s6)
        d2 = self.d2(d1, s5)
        d3 = self.d3(d2, s4)
        d4 = self.d4(d3, s3)
        d5 = self.d5(d4, s2)
        d6 = self.d6(d5, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d6))  # check sigmoid vs tanh
        return logits, rec, encoding

class Unet4_ResidualScorer(Project_DFD_model):
    """
        U-net 4 with Resiudal Encoder + Scorer, 5 encoders and 5 decoders
    """

    def __init__(self, n_classes= 10, large_encoding = True):
        super(Unet4_ResidualScorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS     # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        # self.bottleneck_size = int(self.feature_maps(4)*(self.width/8)*(self.width/8))  (224x224 case)
        self.bottleneck_size = int(self.feature_maps(4)*math.floor(self.width/16)**2) 
        self.residual2conv = False  # if False identity shortcuts are connected after the pooling layer
        self.n_levels = 4
        self.large_encoding = large_encoding
        
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
    
    def _createNet(self):
        # encoder
        self.e1 = Encoder_block_residual(self.n_channels, self.feature_maps(0), self.residual2conv)
        self.e2 = Encoder_block_residual(self.feature_maps(0) , self.feature_maps(1), self.residual2conv)
        self.e3 = Encoder_block_residual(self.feature_maps(1) , self.feature_maps(2), self.residual2conv)
        self.e4 = Encoder_block_residual(self.feature_maps(2) , self.feature_maps(3), self.residual2conv)
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(3) , self.feature_maps(4))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
        
        # self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/16))
        # self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/16))
        # self.fc2 = nn.Linear(int(self.bottleneck_size/16), int(self.bottleneck_size/64))
        # self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        # self.fc3 = nn.Linear(int(self.bottleneck_size/64), self.n_classes)
        
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        # decoder 
        self.d1 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d2 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d3 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d4 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()
            
        # self.model = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x, verbose = False):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
    
        # bottleneck (encoding)
        bottleneck = self.b(p4)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)
        
        # classification        
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)
        
        
        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")
        
        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1
        
        # decoder 
        d1 = self.d1(bottleneck, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d4))  # check sigmoid vs tanh
        return logits, rec, encoding

class Unet5_ResidualScorer(Project_DFD_model):
    """
        U-net 5 with Resiudal Encoder + Scorer, 5 encoders (residual) and 5 decoders
    """
    
    def __init__(self, n_classes= 10, large_encoding = True):
        super(Unet5_ResidualScorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.bottleneck_size = int(self.feature_maps(5)*math.floor(self.width/32)**2) 
        self.residual2conv = False  # if False identity shortcuts are connected after the pooling layer
        self.n_levels = 5
        self.large_encoding = large_encoding
        
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
    
    def _createNet(self):
        
        # encoder
        self.e1 = Encoder_block_residual(self.n_channels, self.feature_maps(0), self.residual2conv)
        self.e2 = Encoder_block_residual(self.feature_maps(0) , self.feature_maps(1), self.residual2conv)
        self.e3 = Encoder_block_residual(self.feature_maps(1) , self.feature_maps(2), self.residual2conv)
        self.e4 = Encoder_block_residual(self.feature_maps(2) , self.feature_maps(3), self.residual2conv)
        self.e5 = Encoder_block_residual(self.feature_maps(3) , self.feature_maps(4), self.residual2conv, kernel_size=2) # 112x112 addition
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(4) , self.feature_maps(5))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
        
        # self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/16))
        # self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/16))
        # self.fc2 = nn.Linear(int(self.bottleneck_size/16), int(self.bottleneck_size/64))
        # self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        # self.fc3 = nn.Linear(int(self.bottleneck_size/64), self.n_classes)
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        
        # decoder 
        
        # define conditions for padding 
        c = INPUT_WIDTH/(math.pow(2,self.n_levels))
        
        if c%1!=0:
            self.d1 = Decoder_block(self.feature_maps(5) , self.feature_maps(4), out_pad=1) # 112x112 addition
        else:
            self.d1 = Decoder_block(self.feature_maps(5) , self.feature_maps(4))
        self.d2 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d3 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d4 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d5 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()
            
        # self.model = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x, verbose = False):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
    
        # bottleneck (encoding)
        bottleneck  = self.b(p5)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)

        
        # classification
        
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)
        
        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")
        
        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1
        # decoder 
        d1 = self.d1(bottleneck, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d5))  # check sigmoid vs tanh
        return logits, rec, encoding
    
class Unet6_ResidualScorer(Project_DFD_model):
    """
        U-net 6 + Scorer, 6 encoders (residual) and 6 decoders.
        This version include an additional layer for the scorer
    """

    def __init__(self, n_classes = 10, large_encoding = True):
        super(Unet6_ResidualScorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        # self.bottleneck_size = int(self.feature_maps(6)*(w/64)*(w/64))
        self.bottleneck_size = int(self.feature_maps(6)*math.floor(self.width/64)**2)  # if 224x224 the width is divided by 32
        self.residual2conv = False  # if False identity shortcuts are connected after the pooling layer
        self.n_levels = 6
        self.large_encoding = large_encoding
        
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
    
    def _createNet(self):
        
        # encoder
        self.e1 = Encoder_block_residual(self.n_channels, self.feature_maps(0), self.residual2conv)
        self.e2 = Encoder_block_residual(self.feature_maps(0) , self.feature_maps(1), self.residual2conv)
        self.e3 = Encoder_block_residual(self.feature_maps(1) , self.feature_maps(2), self.residual2conv)
        self.e4 = Encoder_block_residual(self.feature_maps(2) , self.feature_maps(3), self.residual2conv)
        self.e5 = Encoder_block_residual(self.feature_maps(3) , self.feature_maps(4), self.residual2conv, kernel_size=2) # 112x112 addition
        self.e6 = Encoder_block_residual(self.feature_maps(4) , self.feature_maps(5), self.residual2conv, kernel_size=2)
    
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(5) , self.feature_maps(6))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
        
        # series of FC layers
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        # decoder 
        
        # define conditions for padding 
        c1 = INPUT_WIDTH/(math.pow(2,self.n_levels))
        c2 = INPUT_WIDTH/(math.pow(2,self.n_levels-1))
        
        if c1%1!=0:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5), out_pad=1) # for odd spatial dimensions
        else:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5)) # for odd spatial dimensions
        if c2%1!=0:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4), out_pad=1) # 112x112 addition
        else:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4)) # 112x112 addition
        self.d3 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d4 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d5 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d6 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()

    def forward(self, x, verbose = False):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        s6, p6 = self.e6(p5)
    
        # bottleneck (encoding)
        bottleneck  = self.b(p6)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)

        # classification
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)
        
        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")
        
        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1

        d1 = self.d1(bottleneck, s6)
        d2 = self.d2(d1, s5)
        d3 = self.d3(d2, s4)
        d4 = self.d4(d3, s3)
        d5 = self.d5(d4, s2)
        d6 = self.d6(d5, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d6))  # check sigmoid vs tanh
        return logits, rec, encoding

class Unet6L_ResidualScorer(Project_DFD_model):
    """
        U-net 6 + Scorer, 6 encoders (residual) and 6 decoders.
        This version include an additional layer for the scorer and LargeConv_block instead of Conv_blocks
    """

    def __init__(self, n_classes= 10, large_encoding = True):
        super(Unet6L_ResidualScorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        # self.bottleneck_size = int(self.feature_maps(6)*(w/64)*(w/64))
        self.bottleneck_size = int(self.feature_maps(6)*math.floor(self.width/64)**2)  # if 224x224 the width is divided by 32
        self.residual2conv = False  # if False identity shortcuts are connected after the pooling layer
        self.n_levels = 6
        self.large_encoding = large_encoding
        
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
    
    def _createNet(self):
        
        # encoder
        self.e1 = LargeEncoder_block_residual(self.n_channels, self.feature_maps(0), self.residual2conv)
        self.e2 = LargeEncoder_block_residual(self.feature_maps(0) , self.feature_maps(1), self.residual2conv)
        self.e3 = LargeEncoder_block_residual(self.feature_maps(1) , self.feature_maps(2), self.residual2conv)
        self.e4 = LargeEncoder_block_residual(self.feature_maps(2) , self.feature_maps(3), self.residual2conv)
        self.e5 = LargeEncoder_block_residual(self.feature_maps(3) , self.feature_maps(4), self.residual2conv, kernel_size=2) # 112x112 addition
        self.e6 = LargeEncoder_block_residual(self.feature_maps(4) , self.feature_maps(5), self.residual2conv, kernel_size=2)
    
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(5) , self.feature_maps(6))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
        
        # series of FC layers
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        # decoder 
        
        # define conditions for padding 
        c1 = INPUT_WIDTH/(math.pow(2,self.n_levels))
        c2 = INPUT_WIDTH/(math.pow(2,self.n_levels-1))
        
        if c1%1!=0:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5), out_pad=1) # for odd spatial dimensions
        else:
            self.d1 = Decoder_block(self.feature_maps(6) , self.feature_maps(5)) # for odd spatial dimensions
        if c2%1!=0:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4), out_pad=1) # 112x112 addition
        else:
            self.d2 = Decoder_block(self.feature_maps(5) , self.feature_maps(4)) # 112x112 addition
        self.d3 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d4 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d5 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d6 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()

    def forward(self, x, verbose = False ):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        s6, p6 = self.e6(p5)
    
        # bottleneck (encoding)
        bottleneck  = self.b(p6)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)

        # classification
        
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)
        
        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")
        
        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1

        d1 = self.d1(bottleneck, s6)
        d2 = self.d2(d1, s5)
        d3 = self.d3(d2, s4)
        d4 = self.d4(d3, s3)
        d5 = self.d5(d4, s2)
        d6 = self.d6(d5, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d6))  # check sigmoid vs tanh
        return logits, rec, encoding

#                                       custom Unet + Confidence
class Unet4_Scorer_Confidence(Project_DFD_model):
    """
        U-net 4 + Scorer, 4 encoders and 4 decoders + cponfidence estimantion.
        Confidence reflects the model's abiliy to produce a correct prediction for any given input
    """

    def __init__(self, n_classes= 10,  large_encoding = True):
        super(Unet4_Scorer_Confidence,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes,)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        # self.bottleneck_size = int(self.feature_maps(4)*(self.width/16)*(self.width/16))
        self.bottleneck_size = int(self.feature_maps(4)*math.floor(self.width/16)**2) 
        self.n_levels = 4
        self.large_encoding = large_encoding
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal_module()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
        self._init_weights_normal_module(self.conf_layer)
    
    def _createNet(self):
        # encoder
        self.e1 = Encoder_block(self.n_channels, self.feature_maps(0))
        self.e2 = Encoder_block(self.feature_maps(0) , self.feature_maps(1))
        self.e3 = Encoder_block(self.feature_maps(1) , self.feature_maps(2))
        self.e4 = Encoder_block(self.feature_maps(2) , self.feature_maps(3))
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(3) , self.feature_maps(4))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
        
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        # confidence branch
        self.sigmoid = nn.Sigmoid()
        self.conf_layer = nn.Linear(int(self.bottleneck_size/128), 1)
        
        
        # decoder 
        self.d1 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d2 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d3 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d4 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()
            
        # self.model = nn.Sequential(self.encoder, self.decoder)


    def forward(self, x, verbose = False):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
    
        # bottleneck (encoding)
        bottleneck = self.b(p4)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)

        
        # classification
        
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)
        
        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")
        
        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1
        
        # confidence logit
        confidence = self.sigmoid(self.conf_layer(c4))
        
        
        # decoder 
        d1 = self.d1(bottleneck, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d4))  # check sigmoid vs tanh
        return logits, rec, encoding, confidence

class Unet5_Scorer_Confidence(Project_DFD_model):
    """
        U-net 5 + Scorer + Confidence, 5 encoders and 5 decoders
    """

    def __init__(self, n_classes= 10, large_encoding = True):
        super(Unet5_Scorer_Confidence,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.bottleneck_size = int(self.feature_maps(5)*math.floor(self.width/32)**2)  
        self.n_levels = 5
        self.large_encoding = large_encoding       
        
        # create net and initialize
        self._createNet()
        
        # initialize conv layers
        self._init_weights_kaimingNormal()
        
        # initialize FC layer
        self._init_weights_normal_module(self.fc1)
        self._init_weights_normal_module(self.fc2)
        self._init_weights_normal_module(self.fc3)
        self._init_weights_normal_module(self.fc4)
        self._init_weights_normal_module(self.fc5)
    
    def _createNet(self):
        
        # encoder
        self.e1 = Encoder_block(self.n_channels, self.feature_maps(0))
        self.e2 = Encoder_block(self.feature_maps(0) , self.feature_maps(1))
        self.e3 = Encoder_block(self.feature_maps(1) , self.feature_maps(2))
        self.e4 = Encoder_block(self.feature_maps(2) , self.feature_maps(3))
        self.e5 = Encoder_block(self.feature_maps(3) , self.feature_maps(4))
        
        # bottlenech (encoding)
        self.b = Conv_block(self.feature_maps(4) , self.feature_maps(5))
        
        # Flatten the encoding
        self.flatten = nn.Flatten()
        
        # classification branch
        self.do     = nn.Dropout(p=0.3)
        self.relu   = nn.ReLU()
                
        self.fc1 = nn.Linear(self.bottleneck_size, int(self.bottleneck_size/8))
        self.bn1 = nn.BatchNorm1d(int(self.bottleneck_size/8))
        self.fc2 = nn.Linear(int(self.bottleneck_size/8), int(self.bottleneck_size/32))
        self.bn2 = nn.BatchNorm1d(int(self.bottleneck_size/32))
        self.fc3 = nn.Linear(int(self.bottleneck_size/32), int(self.bottleneck_size/64))
        self.bn3 = nn.BatchNorm1d(int(self.bottleneck_size/64))
        self.fc4 = nn.Linear(int(self.bottleneck_size/64), int(self.bottleneck_size/128))
        self.bn4 = nn.BatchNorm1d(int(self.bottleneck_size/128))
        self.fc5 = nn.Linear(int(self.bottleneck_size/128), int(self.n_classes))
        
        # confidence branch
        self.sigmoid = nn.Sigmoid()
        self.conf_layer = nn.Linear(int(self.bottleneck_size/128), 1)
        
        
        # decoder 
        
        # define conditions for padding, considering downsampling of odd dimensions
        c = INPUT_WIDTH/(math.pow(2,self.n_levels))
        
        if c%1!=0:
            self.d1 = Decoder_block(self.feature_maps(5) , self.feature_maps(4), out_pad=1) # 112x112 addition
        else:
            self.d1 = Decoder_block(self.feature_maps(5) , self.feature_maps(4))
        self.d2 = Decoder_block(self.feature_maps(4) , self.feature_maps(3))
        self.d3 = Decoder_block(self.feature_maps(3) , self.feature_maps(2))
        self.d4 = Decoder_block(self.feature_maps(2) , self.feature_maps(1))
        self.d5 = Decoder_block(self.feature_maps(1) , self.feature_maps(0))
            
        # self.out= decoder_block(64, self.n_channels)
        self.out = nn.Conv2d(self.feature_maps(0), self.n_channels, kernel_size=1, padding=0)
        self.decoder_out_fn = nn.Sigmoid()
            
        # self.model = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x, verbose = False):
        """
            Returns: logits, reconstruction, encoding
        """
        
        # encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
    
        # bottleneck (encoding)
        bottleneck  = self.b(p5)
        enc         = self.flatten(bottleneck)
        
        if verbose: print ("enc shape ", enc.shape)
        
        # classification
        c1          = self.relu(self.bn1(self.fc1(enc)))
        c1d         = self.do(c1)
        
        c2          = self.relu(self.bn2(self.fc2(c1d)))
        c2d         = self.do(c2)
        
        c3          = self.relu(self.bn3(self.fc3(c2d)))
        c3d         = self.do(c3)
        
        c4          = self.relu(self.bn4(self.fc4(c3d)))
        c4d         = self.do(c4)
        
        logits      = self.fc5(c4d)
        
        
        if verbose:
            print("c1 shape ", c1.shape, "\n", "c2 shape ", c2.shape, "\n", "c3 shape ", c3.shape, "\n","c4 shape ", c4.shape, "\n")
            
        # select large or small encoding for the forward 
        if self.large_encoding:
            encoding = enc
        else:
            encoding = c1
        
        
        # confidence logit
        confidence = self.sigmoid(self.conf_layer(c4))
        
        # decoder 
        d1 = self.d1(bottleneck, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d5))  # check sigmoid vs tanh
        return logits, rec, encoding, confidence


#                                       custom abnormality modules

# 2nd models superclass for OOD detection
class Project_abnorm_model(nn.Module): 
    def __init__(self):
        super(Project_abnorm_model, self).__init__()
        print("Initializing {} ...".format(self.__class__.__name__))

    def _createNet(self):
        raise NotImplementedError
            
    def _init_weights_normal(self, model = None):
        # Initialize the weights with Gaussian distribution
        print(f"Weights initialization using Gaussian distribution")
        
        if model is None: model = self
        for param in model.parameters():
            if len(param.shape) > 1:
                T.nn.init.normal_(param, mean=0, std=0.01) 
                 
    def getSummary(self, input_shape = None, verbose = True):
        
        """
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this type -> color,width,height
        """
        
        
        if input_shape is None:
            input_shape = self.input_shape
            
        try:
            model_stats = summary(self, input_shape, verbose = int(verbose))
            return str(model_stats)
        except:            
            summ = ""
            n_params = 0
            for k,v in self.getLayers().items():
                summ += "{:<30} -> {:<30}".format(k,str(tuple(v.shape))) + "\n"
                n_params += T.numel(v)
            summ += "Total number of parameters: {}\n".format(n_params)
            if verbose: print(summ)
            return summ
        
    def to_device(self, device):
        self.to(device)
    
    def getLayers(self):
        return dict(self.named_parameters())
    
    def getAttributes(self):
        att = self.__dict__
        
        def valid(pair):
            # unzip pair k,v
            _, x = pair 
            type_x = type(x)
            condition =  (type_x is int) or (type_x is str) or (type_x is tuple)
            return condition
        
        filterd_att = dict(filter(valid, att.items()))
        return filterd_att
    
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
    
class Abnormality_module_Basic(Project_abnorm_model):
    
    """ problems withs model:
        - high computational cost (no reduction from image elaborated data)
        - high difference of values between softmax probabilites, encoding, and residual flatten vector    
    """
    
    def __init__(self, shape_softmax_probs, shape_encoding, shape_residual):        
        super().__init__()
        self.probs_softmax_shape   = shape_softmax_probs
        self.encoding_shape = shape_encoding
        self.residual_shape = shape_residual
        self.input_shape = (shape_softmax_probs, shape_encoding, shape_residual)   # input_shape doesn't consider the batch
        
        self._createNet()
        self._init_weights_normal()
    
    def _createNet(self):
        # print(self.probs_softmax_shape)
        # print(self.encoding_shape)
        # print(self.residual_shape)
        tot_features_0          = self.probs_softmax_shape[1] + self.encoding_shape[1] + (self.residual_shape[1]*self.residual_shape[2]* self.residual_shape[3])
        # print(tot_features_0)
        
        first_reduction_coeff   = 1000
        
        
        if int(tot_features_0/first_reduction_coeff) > 4096:
            tot_features_1      = int(tot_features_0/first_reduction_coeff)
        else:
            tot_features_1      = 4096
            
        tot_features_2      = 1024
        
        
        # taken from official work 
        tot_feaures_risk_1  = 512
        tot_feaures_risk_2  = 128
        tot_feaures_final   = 1
        
        self.gelu = T.nn.GELU()
        self.sigmoid = T.nn.Sigmoid()
        
        # preliminary layers
        self.fc1 = T.nn.Linear(tot_features_0,tot_features_1)
        self.bn1 = T.nn.BatchNorm1d(tot_features_1)
        self.fc2 = T.nn.Linear(tot_features_1,tot_features_2)
        self.bn2 = T.nn.BatchNorm1d(tot_features_2)
        
        # risk section
        self.fc_risk_1      = T.nn.Linear(tot_features_2,tot_feaures_risk_1)
        self.bn_risk_1      = T.nn.BatchNorm1d(tot_feaures_risk_1)
        self.fc_risk_2      = T.nn.Linear(tot_feaures_risk_1,tot_feaures_risk_2)
        self.bn_risk_2      = T.nn.BatchNorm1d(tot_feaures_risk_2)
        self.fc_risk_final  = T.nn.Linear(tot_feaures_risk_2,tot_feaures_final)
        self.bn_risk_final  = T.nn.BatchNorm1d(tot_feaures_final)
        
    def forward(self, probs_softmax, encoding, residual, verbose = False):
        
                
        # flat the residual
        flatten_residual = T.flatten(residual, start_dim=1)
        
        # build the vector input 
        x = T.cat((probs_softmax, encoding, flatten_residual), dim = 1)
        if verbose: print("input module b shape -> ", x.shape)
        
        # print(x.shape)
        
        # preliminary layers
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.gelu(self.bn2(self.fc2(x)))
        
        # risk section
        x = self.gelu(self.bn_risk_1(self.fc_risk_1(x)))
        x = self.gelu(self.bn_risk_2(self.fc_risk_2(x)))
        x = self.gelu(self.bn_risk_final(self.fc_risk_final(x)))
        
        return x

class Abnormality_module_Encoder_v1(Project_abnorm_model):
    
    """ 
        based on Abnormality_module_Basic:
        - reduction of computational cost from the squared residual using conv encoder.
    """
    
    def __init__(self, shape_softmax_probs, shape_encoding, shape_residual):
        super().__init__()
        
        self.probs_softmax_shape   = shape_softmax_probs
        self.encoding_shape = shape_encoding
        self.residual_shape = shape_residual
        self.input_shape = (shape_softmax_probs, shape_encoding, shape_residual)   # input_shape doesn't consider the batch
        self._createNet()
        self._init_weights_normal()
    
    def _createNet(self):
        residual_length  = self.encoding_shape[1]  # flatten residual should have same dim of the encoding
        # tot_features_0          = self.shape_softmax_probs[1] + self.shape_encoding[1] + self.shape_residual[1]
        tot_features_0          = self.probs_softmax_shape[1] +  self.encoding_shape[1] + residual_length


        # conv section
        
        self.e1 = Encoder_block(in_c =  self.residual_shape[1] , out_c = 16)
        self.e2 = Encoder_block(in_c =  16, out_c = 32)
        self.e3 = Encoder_block(in_c =  32, out_c = 64)
        # e2 = Encoder_block(in_c =  64, out_c = 128)
        
        tot_features_1      = 2048
        tot_features_2      = 1024
        
        # # taken from official work 
        tot_feaures_risk_1  = 512
        tot_feaures_risk_2  = 128
        tot_feaures_final   = 1
        
        self.gelu = T.nn.GELU()
        self.sigmoid = T.nn.Sigmoid()
        
        # preliminary layers
        self.fc1 = T.nn.Linear(tot_features_0,tot_features_1)
        self.bn1 = T.nn.BatchNorm1d(tot_features_1)
        self.fc2 = T.nn.Linear(tot_features_1,tot_features_2)
        self.bn2 = T.nn.BatchNorm1d(tot_features_2)
        
        # risk section
        self.fc_risk_1      = T.nn.Linear(tot_features_2,tot_feaures_risk_1)
        self.bn_risk_1      = T.nn.BatchNorm1d(tot_feaures_risk_1)
        self.fc_risk_2      = T.nn.Linear(tot_feaures_risk_1,tot_feaures_risk_2)
        self.bn_risk_2      = T.nn.BatchNorm1d(tot_feaures_risk_2)
        self.fc_risk_final  = T.nn.Linear(tot_feaures_risk_2,tot_feaures_final)
        self.bn_risk_final  = T.nn.BatchNorm1d(tot_feaures_final)
        
    def forward(self, probs_softmax, encoding, residual, verbose = False):
        
        
        # conv forward for the 
        _, residual_out = self.e1(residual)
        _, residual_out = self.e2(residual_out)
        _, residual_out = self.e3(residual_out)
        
        # flat the residual
        flatten_residual = T.flatten(residual_out, start_dim=1)
        
        # build the vector input 
        x = T.cat((probs_softmax, encoding, flatten_residual), dim = 1)
        if verbose: print("input module b shape -> ", x.shape)
        
        # preliminary layers
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.gelu(self.bn2(self.fc2(x)))
        
        # risk section
        x = self.gelu(self.bn_risk_1(self.fc_risk_1(x)))
        x = self.gelu(self.bn_risk_2(self.fc_risk_2(x)))
        x = self.gelu(self.bn_risk_final(self.fc_risk_final(x)))
        
        return x

class Encoder_block_v2(nn.Module):
    """ 
    it reduces the spatial dimensionality of half
    respect the Encoder_block class, downsampling is performed through conv layer instead of pooling one.
    """ 
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride = 1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride = 2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=3, stride = 1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_c)
        # self.relu = nn.ReLU()
        self.gelu = T.nn.GELU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)
        return x

class Abnormality_module_Encoder_v2(Project_abnorm_model):
    
    """ 
        based on Abnormality_module_Encoder_v1:
        - it substitue the Encoder_block (conv + pooling) with Encoder_block_v2 (just conv)
        - it increases the number of neurons in the net
        - add one fc layer 
    """
    
    def __init__(self, shape_softmax_probs, shape_encoding, shape_residual):
        super().__init__()
        
        self.probs_softmax_shape   = shape_softmax_probs
        self.encoding_shape = shape_encoding
        self.residual_shape = shape_residual
        self.input_shape = (shape_softmax_probs, shape_encoding, shape_residual)   # input_shape doesn't consider the batch
        self._createNet()
        self._init_weights_normal()
    
    def _createNet(self):
        residual_length  = self.encoding_shape[1]  # flatten residual should have same dim of the encoding
        # tot_features_0          = self.shape_softmax_probs[1] + self.shape_encoding[1] + self.shape_residual[1]
        tot_features_0          = self.probs_softmax_shape[1] +  self.encoding_shape[1] + residual_length


        # conv section
        
        self.e1 = Encoder_block_v2(in_c =  self.residual_shape[1] , out_c = 16)
        self.e2 = Encoder_block_v2(in_c =  16, out_c = 32)
        self.e3 = Encoder_block_v2(in_c =  32, out_c = 64)
        # e2 = Encoder_block(in_c =  64, out_c = 128)
        
        tot_features_1      = 4096
        tot_features_2      = 2048
        tot_features_3      = 1024
        
        
        # # taken from official work 
        tot_feaures_risk_1  = 512
        tot_feaures_risk_2  = 128
        tot_feaures_final   = 1
        
        self.gelu = T.nn.GELU()
        self.sigmoid = T.nn.Sigmoid()
        
        # preliminary layers
        self.fc1 = T.nn.Linear(tot_features_0,tot_features_1)
        self.bn1 = T.nn.BatchNorm1d(tot_features_1)
        self.fc2 = T.nn.Linear(tot_features_1,tot_features_2)
        self.bn2 = T.nn.BatchNorm1d(tot_features_2)
        self.fc3 = T.nn.Linear(tot_features_2,tot_features_3)
        self.bn3 = T.nn.BatchNorm1d(tot_features_3)
        
        # risk section
        self.fc_risk_1      = T.nn.Linear(tot_features_3,tot_feaures_risk_1)
        self.bn_risk_1      = T.nn.BatchNorm1d(tot_feaures_risk_1)
        self.fc_risk_2      = T.nn.Linear(tot_feaures_risk_1,tot_feaures_risk_2)
        self.bn_risk_2      = T.nn.BatchNorm1d(tot_feaures_risk_2)
        self.fc_risk_final  = T.nn.Linear(tot_feaures_risk_2,tot_feaures_final)
        self.bn_risk_final  = T.nn.BatchNorm1d(tot_feaures_final)
        
    def forward(self, probs_softmax, encoding, residual, verbose = False):
        
        # conv forward for the 
        residual_out = self.e1(residual)
        residual_out = self.e2(residual_out)
        residual_out = self.e3(residual_out)

        # flat the residual
        flatten_residual = T.flatten(residual_out, start_dim=1)
        
        # build the vector input 
        x = T.cat((probs_softmax, encoding, flatten_residual), dim = 1)
        if verbose: print("input module b shape -> ", x.shape)
        
        # preliminary layers
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.gelu(self.bn2(self.fc2(x)))
        x = self.gelu(self.bn3(self.fc3(x)))
        
        # risk section
        x = self.gelu(self.bn_risk_1(self.fc_risk_1(x)))
        x = self.gelu(self.bn_risk_2(self.fc_risk_2(x)))
        x = self.gelu(self.bn_risk_final(self.fc_risk_final(x)))
        
        return x

class Abnormality_module_Encoder_v3(Project_abnorm_model):
    
    """ 
        based on Abnormality_module_Encoder_v3:
        - use a smaleller encoding as input (from fc layes of the scorer)
        - higher reduction for the residual before the fc layers (more encoding blocks)
        - consequent reduced n. params for the final fc risk section.
        
        **requires** output from a classifier with: classifier_name.large_encoding = False
    """
    
    def __init__(self, shape_softmax_probs, shape_encoding, shape_residual):
        super().__init__()
        
        self.probs_softmax_shape   = shape_softmax_probs
        self.encoding_shape = shape_encoding
        self.residual_shape = shape_residual
        self.input_shape = (shape_softmax_probs, shape_encoding, shape_residual)   # input_shape doesn't consider the batch
        self._createNet()
        self._init_weights_normal()
    
    def _createNet(self):
        
        residual_length  = self.encoding_shape[1]  # flatten residual (after encoding) should have same dim of the encoding
        tot_features_0   = self.probs_softmax_shape[1] +  self.encoding_shape[1] + residual_length

        # conv section
        self.e1 = Encoder_block_v2(in_c =  self.residual_shape[1] , out_c = 4)
        self.e2 = Encoder_block_v2(in_c =  4, out_c = 8)
        self.e3 = Encoder_block_v2(in_c =  8, out_c = 16)
        self.e4 = Encoder_block_v2(in_c =  16, out_c = 32)

        tot_features_1      = 1024       
         
        # taken from official work 
        tot_feaures_risk_1  = 512
        tot_feaures_risk_2  = 128
        tot_feaures_final   = 1
        
        self.gelu = T.nn.GELU()
        self.sigmoid = T.nn.Sigmoid()
        
        # preliminary layers
        self.fc1 = T.nn.Linear(tot_features_0,tot_features_1)
        self.bn1 = T.nn.BatchNorm1d(tot_features_1)
        
        # risk section
        self.fc_risk_1      = T.nn.Linear(tot_features_1,tot_feaures_risk_1)
        self.bn_risk_1      = T.nn.BatchNorm1d(tot_feaures_risk_1)
        self.fc_risk_2      = T.nn.Linear(tot_feaures_risk_1,tot_feaures_risk_2)
        self.bn_risk_2      = T.nn.BatchNorm1d(tot_feaures_risk_2)
        self.fc_risk_final  = T.nn.Linear(tot_feaures_risk_2,tot_feaures_final)
        self.bn_risk_final  = T.nn.BatchNorm1d(tot_feaures_final)
        
    def forward(self, probs_softmax, encoding, residual, verbose = False):
        
        # for 112p images
        # conv forward for the  
        residual_out = self.e1(residual)
        residual_out = self.e2(residual_out)
        residual_out = self.e3(residual_out)
        residual_out = self.e4(residual_out)   # 64, 7, 7 shape
        
        print(residual_out.shape)
        
        # flat the residual
        flatten_residual = T.flatten(residual_out, start_dim=1)
        
        # build the vector input 
        x = T.cat((probs_softmax, encoding, flatten_residual), dim = 1)
        if verbose: print("input module b shape -> ", x.shape)
        
        # preliminary layers
        x = self.gelu(self.bn1(self.fc1(x)))
        
        # risk section
        x = self.gelu(self.bn_risk_1(self.fc_risk_1(x)))
        x = self.gelu(self.bn_risk_2(self.fc_risk_2(x)))
        x = self.gelu(self.bn_risk_final(self.fc_risk_final(x)))
        
        return x
    
class Abnormality_module_Encoder_v4(Project_abnorm_model):
    
    """ 
        based on Abnormality_module_Encoder_v2:
        - moves the concatenation after several fc layer separately defined for encoding and residual
    """
    
    def __init__(self, shape_softmax_probs, shape_encoding, shape_residual):
        super().__init__()
        
        self.probs_softmax_shape   = shape_softmax_probs
        self.encoding_shape = shape_encoding
        self.residual_shape = shape_residual
        self.input_shape = (shape_softmax_probs, shape_encoding, shape_residual)   # input_shape doesn't consider the batch
        self._createNet()
        self._init_weights_normal()
    
    def _createNet(self):
        residual_length  = self.encoding_shape[1]  # flatten residual should have same dim of the encoding
        

        tot_feaures_risk_3          = 10  # 10 features for encoding and ten for residual before last layer
        tot_feature_risk_concat     =  self.probs_softmax_shape[1] + tot_feaures_risk_3*2
        
        tot_features_1      = 4096
        tot_features_2      = 2048
        tot_features_3      = 1024
        
        # # taken from official work 
        tot_feaures_risk_1  = 512
        tot_feaures_risk_2  = 128

        
        tot_feaures_final   = 1
        
        self.gelu = T.nn.GELU()
        self.sigmoid = T.nn.Sigmoid()
        
        #                           for Residual 
        # conv section
        self.e1 = Encoder_block_v2(in_c =  self.residual_shape[1] , out_c = 16)
        self.e2 = Encoder_block_v2(in_c =  16, out_c = 32)
        self.e3 = Encoder_block_v2(in_c =  32, out_c = 64)
        

        # preliminary layers after flatterning output residual from conv blocks
        self.fc1r = T.nn.Linear(residual_length,tot_features_1)
        self.bn1r = T.nn.BatchNorm1d(tot_features_1)
        self.fc2r = T.nn.Linear(tot_features_1,tot_features_2)
        self.bn2r = T.nn.BatchNorm1d(tot_features_2)
        self.fc3r = T.nn.Linear(tot_features_2,tot_features_3)
        self.bn3r = T.nn.BatchNorm1d(tot_features_3)
        
        # risk section
        self.fc_risk_1r      = T.nn.Linear(tot_features_3,tot_feaures_risk_1)
        self.bn_risk_1r      = T.nn.BatchNorm1d(tot_feaures_risk_1)
        self.fc_risk_2r      = T.nn.Linear(tot_feaures_risk_1,tot_feaures_risk_2)
        self.bn_risk_2r      = T.nn.BatchNorm1d(tot_feaures_risk_2)
        self.fc_risk_3r      = T.nn.Linear(tot_feaures_risk_2,tot_feaures_risk_3)
        self.bn_risk_3r      = T.nn.BatchNorm1d(tot_feaures_risk_3)
        
        #                           for encoding 

        # preliminary layers
        self.fc1e = T.nn.Linear(self.encoding_shape[1],tot_features_1)
        self.bn1e = T.nn.BatchNorm1d(tot_features_1)
        self.fc2e = T.nn.Linear(tot_features_1,tot_features_2)
        self.bn2e = T.nn.BatchNorm1d(tot_features_2)
        self.fc3e = T.nn.Linear(tot_features_2,tot_features_3)
        self.bn3e = T.nn.BatchNorm1d(tot_features_3)
        
        # risk section
        self.fc_risk_1e      = T.nn.Linear(tot_features_3,tot_feaures_risk_1)
        self.bn_risk_1e      = T.nn.BatchNorm1d(tot_feaures_risk_1)
        self.fc_risk_2e      = T.nn.Linear(tot_feaures_risk_1,tot_feaures_risk_2)
        self.bn_risk_2e      = T.nn.BatchNorm1d(tot_feaures_risk_2)
        self.fc_risk_3e      = T.nn.Linear(tot_feaures_risk_2,tot_feaures_risk_3)
        self.bn_risk_3e      = T.nn.BatchNorm1d(tot_feaures_risk_3)
        
        # final fc layer after contenation 
        self.fc_risk_final  = T.nn.Linear(tot_feature_risk_concat,tot_feaures_final)
        self.bn_risk_final  = T.nn.BatchNorm1d(tot_feaures_final)
        
    def forward(self, probs_softmax, encoding, residual, verbose = False):
        
        #                       residual branch 
        residual_out = self.e1(residual)
        residual_out = self.e2(residual_out)
        residual_out = self.e3(residual_out)

        # flat the residual
        flatten_residual = T.flatten(residual_out, start_dim=1)
        
        # preliminary layers
        x = self.gelu(self.bn1r(self.fc1r(flatten_residual)))
        x = self.gelu(self.bn2r(self.fc2r(x)))
        x = self.gelu(self.bn3r(self.fc3r(x)))
        
        # risk section
        x = self.gelu(self.bn_risk_1r(self.fc_risk_1r(x)))
        x = self.gelu(self.bn_risk_2r(self.fc_risk_2r(x)))
        x_r = self.gelu(self.bn_risk_3r(self.fc_risk_3r(x)))

        
        #                       encoding branch 
        # preliminary layers
        x = self.gelu(self.bn1e(self.fc1e(encoding)))
        x = self.gelu(self.bn2e(self.fc2e(x)))
        x = self.gelu(self.bn3e(self.fc3e(x)))
        
        # risk section
        x = self.gelu(self.bn_risk_1e(self.fc_risk_1e(x)))
        x = self.gelu(self.bn_risk_2e(self.fc_risk_2e(x)))
        x_e = self.gelu(self.bn_risk_3e(self.fc_risk_3e(x)))
        
        
        # # build the vector as input of final fc layer
        x_concat = T.cat((probs_softmax, x_e, x_r), dim = 1)
        # if verbose: print("input module b shape -> ", x.shape)
        out = self.gelu(self.bn_risk_final(self.fc_risk_final(x_concat)))
        return out   


#_____________________________________Vision Transformer (ViT)_____________________________________        


# Default transformer model settings:

PATCH_SIZE  = 16
EMB_SIZE    = 1024

# _____________________________ViT base from scratch_______________________________________

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)   # n length of the input sequence

        dots = T.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = T.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT_b_scratch(Project_DFD_model):
    """ Implementation of a classic ViT model (base) for classification """
    
    def __init__(self, *, img_size = INPUT_WIDTH, patch_size = PATCH_SIZE, n_classes = 10, emb_size = EMB_SIZE, n_layers = 6,
                 n_heads = 16, pool = 'cls', in_channels = INPUT_CHANNELS, dropout = 0.1, emb_dropout = 0.1):
        
        """
            mlp_dim (int): dimension of the hidden representation in the feedforward or multilayer perceptron block
        """
        
        
        
        super().__init__(c = in_channels,h = img_size,w = img_size, n_classes = n_classes)
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        
        self.emb_size   = emb_size 
        self.patch_size = patch_size
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.dropout    = dropout
        
        # compute head latent space dimensionality
        dim_head = EMB_SIZE//n_heads
        self.dim_head = dim_head
        
        # compute ff encoding dimensionality
        mlp_dim = EMB_SIZE*2
        self.mlp_dim = mlp_dim
        
        # bind Transformer dropout to the one of the embedding
        emb_dropout = self.dropout  
        

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_size),
            nn.LayerNorm(emb_size),
        )

        self.pos_embedding = nn.Parameter(T.randn(1, num_patches + 1, emb_size))
        self.cls_token = nn.Parameter(T.randn(1, 1, emb_size))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(emb_size, n_layers, n_heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(emb_size, n_classes)
        
        self._init_weights_normal()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = T.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
         
        logits  = self.mlp_head(x)
        
        return logits

# _____________________________ViT base pre-trained  _______________________________________________

def get_vitTimm_models(print_it = True, pretrained = True):
    models = [model for model in timm.list_models(pretrained=pretrained) if "vit_" in model.lower().strip()]
    if print_it:
        print("\ntimm module, available models:\n")
        print_list(models)
    return models
     
class ViT_timm(Project_DFD_model):
    def __init__(self, n_classes = 10, dropout = 0.1, prog_model = 1):
        """_summary_

        Args:
            num_classes (int, optional): _description_. Defaults to 10.
            dropout (int, optional): dropout rate used in attention and MLP layers. Defaults to 0.
            prog_model (int, optional): progressive id to select the model (use getModels for the complete list)
        """
        super(ViT_timm, self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        self.models_avaiable = ['vit_base_patch16_224', 'vit_base_patch16_224.augreg_in21k']
        self.name = self.models_avaiable[prog_model]
        self.model_vit = timm.create_model(model_name=self.name, pretrained=True, num_classes=n_classes, drop_rate=dropout)
        data_config = timm.data.resolve_model_data_config(self.model_vit)
        
        # get trasnform ops to adapt input
        self.transforms = timm.data.create_transform(**data_config, is_training=True)
        
        # print(data_config)
        # info = get_info_timm_model(self.model_name)
        # print(info)
        
        self.emb_size   = 768 
        self.patch_size = 16
        self.n_heads    = 12
        self.n_layers   = 12
        self.dropout    = dropout
        self.dim_head = self.emb_size //self.n_heads

        print(self.transforms)
        
        
    def getModels(self):
        print_list(self.models_avaiable)
    
    def forward(self, x):
        # Pass the input through the ViT model
        output = self.model_vit(x)

        return output

class ViT_b16_ImageNet(Project_DFD_model):
    """ 
    This is a wrap class for pretraiend Vision Transformer b16 use the getModel function to get the nn.module implementation.
    The model expects color images in RGB standard, of size 244x244
    """
    
    
    def __init__(self, n_classes = 10):
        super(ViT_b16_ImageNet,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        # self.weight_name =  ResNet50_Weights.IMAGENET1K_V2  # weights with accuracy 80.858% on ImageNet 
        self.patch_size     = 16
        self.emb_size       = 768 
        self.n_heads        = 12
        self.n_layers       = 12
        self.weight_name = ViT_B_16_Weights.IMAGENET1K_V1
        self._create_net()
        
        
    def _create_net(self):
        # model = models.resnet50(weights= self.weight_name)
        self.model = models.vit_b_16(weights = self.weight_name, progress = True)
        
        # pre-process to adapt image to the pre-trained model
        self.pre_processing =  ViT_B_16_Weights.DEFAULT.transforms(antialias = True)
            
        # edit fully connected layer for the output
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, self.n_classes)
        
        # turn on the fine-tuning
        self.unfreeze()
        
    def getModel(self):
        return self.model
    
    def freeze(self):
        for _ , param in self.model.named_parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        for _ , param in self.model.named_parameters():
            param.requires_grad = True
    
    def forward(self, x):
        
        # reshape for grayscale images, triplicating the color channel
        if len(x.shape)==4 and x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        
        elif len(x.shape)==3 and x.shape[0] == 1:         
            x = x.expand(3, -1, -1)
        
        x = self.pre_processing(x)
        out = self.model(x)
        return out
  
#_____________________________________ OOD custom models: ViT based ___________________

class AutoEncoder(Project_DFD_model):
    def __init__(self):
        super(AutoEncoder, self).__init__(c = 1, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = None)
        self.flc = 32
        self.zdim = 512
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.flc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        # self.sigmoid = nn.Sigmoid()

        self.encoder.add_module("final_convs", nn.Sequential(
            nn.Conv2d(self.flc, self.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*4, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.zdim, kernel_size=8, stride=1, padding=0)
        ))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.zdim, self.flc, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.flc*4, self.flc*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.flc*2, self.flc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.flc, self.flc, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(self.flc, 1, kernel_size=4, stride=2, padding=1)
            # nn.Sigmoid()
        )
        
        self._init_weights_kaimingNormal()

    def forward(self, x):
        x1 = self.encoder(x)
        # print(x1.shape)
        x2 = self.decoder(x1)
        # print(x2.shape)
        return x2

class AutoEncoder_v2(Project_DFD_model):
    def __init__(self):
        super(AutoEncoder_v2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Sigmoid activation for pixel values in [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(Project_DFD_model):
    # suitable loss function BCE, KL divergence or both, or classic regression loss
    
    def __init__(self, image_size=224, latent_dim=100):
        super(VAE, self).__init__(c = 1, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = None)

        self.image_size = image_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256 * (image_size // 16) ** 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * (image_size // 16) ** 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * (image_size // 16) ** 2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = T.exp(0.5 * logvar)
        eps = T.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(z.size(0), 256, (self.image_size // 16), (self.image_size // 16))
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(recon_x, x, mu, logvar):
        
        BCE =  nn.BCELoss(reduction='sum')(recon_x, x)
        KLD = -0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

class ViT_base_EA(Project_DFD_model):
    """ Implementation of a classic ViT model (base) for classification, providing image encoding (E) and attention map (A) """

    def __init__(self, *, img_size = INPUT_WIDTH, patch_size = PATCH_SIZE, n_classes = 10, emb_size = EMB_SIZE, n_layers = 6,
                    n_heads = 16, pool = 'cls', in_channels = INPUT_CHANNELS, dropout = 0.1, emb_dropout = 0.1,
                    encoding_type = "mean"):
        
        """
            mlp_dim (int): dimension of the hidden representation in the feedforward or multilayer perceptron block
            encoding_type (str, optional). Use "mean" whether to encode by levaraging all the input token as a single vector, or "cls" to take in account only [cls] token. Defaults to "mean"
        """
        
    
        super().__init__(c = in_channels,h = img_size,w = img_size, n_classes = n_classes)
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        
        self.emb_size       = emb_size 
        self.patch_size     = patch_size
        self.n_heads        = n_heads
        self.n_layers       = n_layers
        self.dropout        = dropout
        self.encoding_type  = encoding_type
        
        # compute head latent space dimensionality
        dim_head = EMB_SIZE//n_heads
        self.dim_head = dim_head
        
        # compute ff encoding dimensionality
        mlp_dim = EMB_SIZE*2
        self.mlp_dim = mlp_dim
        
        # bind Transformer dropout to the one of the embedding
        emb_dropout = self.dropout  
        

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_size),
            nn.LayerNorm(emb_size),
        )

        self.pos_embedding = nn.Parameter(T.randn(1, num_patches + 1, emb_size))
        self.cls_token = nn.Parameter(T.randn(1, 1, emb_size))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(emb_size, n_layers, n_heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(emb_size, n_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = T.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # prepare encoding
        enc = x.mean(dim = 1) if  self.encoding_type == "mean" else x[: 0]
        
        # flow through MLP head 
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        logits  = self.mlp_head(x)
        
        return logits, enc 
  
class ViT_timm_EA(Project_DFD_model):
    def __init__(self, n_classes = 10, dropout = 0.1, prog_model = 1, encoding_type = "mean", resize_att_map = True, use_attnmap_cls = True):
        """_summary_

        Args:
            num_classes (int, optional): _description_. Defaults to 10.
            dropout (int, optional): dropout rate used in attention and MLP layers. Defaults to 0.
            prog_model (int, optional): progressive id to select the model (use getModels for the complete list)
            resize_att_map(boolean, optinal): select if output attention map should have same dimension of input images, 
            if false patch_size is used when use_attnmap_cls is True, or lenght of tokens-1, when use_attnmap_cls is False. Defaults to True.
            use_attnmap_cls (boolean, optinal). Select to use attention map from first token (cls), or from pathces .Defaults to True.
        """
        super(ViT_timm_EA, self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        self.models_avaiable = [
            'vit_base_patch16_224',
            'vit_base_patch16_224.augreg_in21k',
            'deit_small_distilled_patch16_224',
            'deit_tiny_distilled_patch16_224'
            ]
        self.name = self.models_avaiable[prog_model]
        print(f"Model architecture select in ViT_EA: {self.name}")
        
        self.model_vit = timm.create_model(model_name=self.name, pretrained=True, num_classes=n_classes, drop_rate=dropout)
        """
            Other ViT timm model parameters:
            num_classes: Mumber of classes for classification head. (int)
            global_pool: Type of global pooling for final sequence (String), default is 'token'. Choose btw: 'avg', 'token', 'map'
            class_token: Use class token (boolean), defaults is True
            drop_rate: Head dropout rate. (float), defaults is 0.
            pos_drop_rate: Position embedding dropout rate.(float), defaults is 0.
            attn_drop_rate: Attention dropout rate. (float), defaults is 0.
        """
        
        
        # data trasnformation
        try:
            data_config = timm.data.resolve_model_data_config(self.model_vit.pretrained_cfg)
            print("found transformation for the input use by pre-trained model")
            print_dict(data_config)
            transform_pretrained = timm.data.create_transform(**data_config)
            self.transform = transforms.Compose([t for t in transform_pretrained.transforms if not isinstance(t, transforms.ToTensor)])
        except:
            print("Not found transformation for the input use by pre-trained model")
            self.transform = None

        self.resize_att_map         = resize_att_map
        self.use_attnmap_cls        = use_attnmap_cls
        
        if prog_model in [0,1]:
            self.emb_size               = 768 
            self.mlp_dim                = 3072
            self.patch_size             = 16
            self.n_heads                = 12
            self.n_layers               = 12
            self.dim_head               = self.emb_size //self.n_heads
        elif prog_model == 2:
            self.emb_size               = 384 
            self.mlp_dim                = "empty"
            self.patch_size             = 16
            self.n_heads                = 6
            self.n_layers               = 12
            self.dim_head               = self.emb_size //self.n_heads
            
        elif prog_model == 3: 
            self.emb_size               = 192 
            self.mlp_dim                = "empty"
            self.patch_size             = 16
            self.n_heads                = 3
            self.n_layers               = 12
            self.dim_head               = self.emb_size //self.n_heads
            
        else:
            self.emb_size               = "empty"
            self.mlp_dim                = "empty"
            self.patch_size             = "empty"
            self.n_heads                = "empty"
            self.n_layers               = "empty"
            self.dim_head               = "empty"
            
        
        self.dropout                = dropout
        self.encoding_type          = encoding_type
        
        # get model parts
        self.embedding  = self.model_vit.patch_embed
        self.encoder    = self.model_vit.blocks
        self.head       = self.model_vit.head    

        # wrapper for attention extraction 
        self.wrapper_prog = 0
        
        if prog_model in [2,3]:
            self.model_vit.blocks[-1].attn.forward = self.forward_wrapper_2(self.model_vit.blocks[-1].attn)
        else:
            if self.wrapper_prog == 0:
                self.model_vit.blocks[-1].attn.forward = self.forward_wrapper(self.model_vit.blocks[-1].attn)
            elif self.wrapper_prog == 1:
                self.model_vit.blocks[-1].attn.forward = self.forward_wrapper_2(self.model_vit.blocks[-1].attn)
        
    def getModels(self):
        print_list(self.models_avaiable)
    
    # reference: https://github.com/lcultrera/WildCapture/
    
    def forward_wrapper(self, attn_obj):
        def forward_wrap(x):
            # get batch, number of elements in the sequence (patches + cls token) and latent representation dims 
            B, N, C = x.shape    # on last layer the input channels has dim self.emb_size (768)
            
            # compute the embedding size for each head
            head_emb_size = C // attn_obj.num_heads

            # get the 3 different features vector for q, k and v (3 in 3rd dimension stands for this) for each head
            # and change dimensions order to: qkv dim, batch, head_dim, sequence_dim, head_embedding
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, head_emb_size).permute(2, 0, 3, 1, 4)
            
            # remove qkv dimension returning the 3 different vectors
            q, k, v = qkv.unbind(dim = 0)

            # print(q.shape)
            # compute the matrix calculus for attention
            
            # first transposition of key vector between sequence_dim, head_embedding, to make matrix multiplication feasible
            k_T = k.transpose(-2, -1)
            
            # matmul + scaling
            attn = (q @ k_T) * attn_obj.scale
            
            # apply softmax over last dimension
            attn = attn.softmax(dim=-1)
            
            # apply dropout
            attn = attn_obj.attn_drop(attn)
            
            # print("att full", attn.shape)
            
            # save the full attention map (used if self.use_attnmap_cls is False)
            attn_obj.attn_map = attn
            
            # get attention map for [cls] token and save, dropping first element in last dimension since is relative to token and not to patches (used if self.use_attnmap_cls is True)
            # attn_obj.cls_attn_map = attn[:, :, 0, 2:]
            attn_obj.cls_attn_map = attn[:, :, 0, 1:]
            
            # compute the remaining operations for the attention forward

            # matmul + exchange of dim between head_dim and sequence_dim
            x = (attn @ v).transpose(1, 2)
            
            # collapse head dim, and head_embedding in a single dimension
            x = x.reshape(B, N, C)
            
            # apply ap and dropout
            x = attn_obj.proj(x)                        # linear activation fuction
            x = attn_obj.proj_drop(x)                 
            
            return x
        return forward_wrap
    
    def forward_wrapper_2(self, attn_obj):
        def forward_wrap(x):
            # get batch, number of elements in the sequence (patches + cls token) and latent representation dims 
            B, N, C = x.shape    # on last layer the input channels has dim self.emb_size (768)
            
            # compute the embedding size for each head
            head_emb_size = C // attn_obj.num_heads

            # get the 3 different features vector for q, k and v (3 in 3rd dimension stands for this) for each head
            # and change dimensions order to: qkv dim, batch, head_dim, sequence_dim, head_embedding
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, head_emb_size).permute(2, 0, 3, 1, 4)
            
            # remove qkv dimension returning the 3 different vectors
            q, k, v = qkv.unbind(dim = 0)

            # print(q.shape)
            # compute the matrix calculus for attention
            
            # first transposition of key vector between sequence_dim, head_embedding, to make matrix multiplication feasible
            k_T = k.transpose(-2, -1)
            
            # matmul + scaling
            attn = (q @ k_T) * attn_obj.scale
            
            # apply softmax over last dimension
            attn = attn.softmax(dim=-1)
            
            # apply dropout
            attn = attn_obj.attn_drop(attn)
            
            # print("att full", attn.shape)
            
            # save the full attention map (used if self.use_attnmap_cls is False)
            attn_obj.attn_map = attn
            
            # get attention map for [cls] token and save, dropping first element in last dimension since is relative to token and not to patches (used if self.use_attnmap_cls is True)
            attn_obj.cls_attn_map = attn[:, :, 0, 2:]
            
            # compute the remaining operations for the attention forward

            # matmul + exchange of dim between head_dim and sequence_dim
            x = (attn @ v).transpose(1, 2)
            
            # collapse head dim, and head_embedding in a single dimension
            x = x.reshape(B, N, C)
            
            # apply ap and dropout
            x = attn_obj.proj(x)                        # linear activation fuction
            x = attn_obj.proj_drop(x)                 
            
            return x
        return forward_wrap
    
    def forward(self, x, verbose = False):
        
        # transform whether available
        if not(self.transform is None):
            x = self.transform(x)

        # print(x.shape)
        features    = self.model_vit.forward_features(x)
        if verbose: print("features shape: ", features.shape)
        
        #                                       1) get encoding
        encoding = features.mean(dim = 1) if self.encoding_type == 'mean' else features[:, 0]
        if verbose: print("encoding shape: ",encoding.shape)

        #                                       2) get logits
        logits      = self.model_vit.forward_head(features)         # using [cls] token embedding 
        if verbose: print("logits shape: ",logits.shape)

        #                                       3) get attention map
    
        if self.use_attnmap_cls:
            att_map     = self.model_vit.blocks[-1].attn.cls_attn_map.mean(dim=1) # mean over heads results
            
            # if use 2nd wrapper, include a value to have a batch of vectors of size: patch_size**2
            if self.wrapper_prog == 1:
                extension_value = T.empty(att_map.shape[0], 1)    # value to be added in the bathes of attention map
                extension_value[:, 0] = att_map[:, 165]
                extension_value = extension_value.cuda()
                att_map = T.cat((att_map, extension_value), dim=1)
            
            att_map     = att_map.view(-1, 14, 14).detach()   # transform in images of dim: patch_size x patch_size
            att_map     = att_map.unsqueeze(dim = 1)          # add channel (grayscale) dim
        else:
            att_map     = self.model_vit.blocks[-1].attn.attn_map.mean(dim=1).detach()
            att_map     = att_map[:, 1:, 1:].view(-1,1,196,196)
        
    
        if self.resize_att_map:
            att_map     = F.interpolate(att_map, (224, 224), mode='bilinear')
            
        return logits, encoding, att_map

#                                       custom abnormality modules

class Abnormality_module_Encoder_VIT_v3(Project_abnorm_model):
    
    """ 
        based on Abnormality_module_Encoder_v3:
        - use a smaleller encoding as input (from fc layes of the scorer)
        - higher reduction for the residual before the fc layers (more encoding blocks)
        - consequent reduced n. params for the final fc risk section.
        
        **requires** output from a classifier with: classifier_name.large_encoding = False
    """
    
    def __init__(self, shape_softmax_probs, shape_encoding, shape_residual):
        super().__init__()
        
        self.probs_softmax_shape   = shape_softmax_probs
        self.encoding_shape = shape_encoding
        self.residual_shape = shape_residual
        self.input_shape = (shape_softmax_probs, shape_encoding, shape_residual)   # input_shape doesn't consider the batch
        self._createNet()
        self._init_weights_normal()
    
    def _createNet(self):
        
        n_encoder_block             = 5
        n_features_latent_residual  = 64
        residual_length  = math.floor(self.residual_shape[2]/(2**n_encoder_block))**2 * n_features_latent_residual # flatten residual (after encoding) should have same dim of the encoding
        
        tot_features_0   = self.probs_softmax_shape[1] +  self.encoding_shape[1] + residual_length

        # conv section, the encoder block must be equal to the number specified in n_encoder_block above
        self.e1 = Encoder_block_v2(in_c =  self.residual_shape[1] , out_c = 4)
        self.e2 = Encoder_block_v2(in_c =  4, out_c = 8)
        self.e3 = Encoder_block_v2(in_c =  8, out_c = 16)
        self.e4 = Encoder_block_v2(in_c =  16, out_c = 32)
        self.e5 = Encoder_block_v2(in_c =  32 , out_c = n_features_latent_residual)

        tot_features_1      = 1024       
         
        # taken from official work 
        tot_feaures_risk_1  = 512
        tot_feaures_risk_2  = 128
        tot_feaures_final   = 1
        
        self.gelu = T.nn.GELU()
        self.sigmoid = T.nn.Sigmoid()
        
        # preliminary layers
        self.fc1 = T.nn.Linear(tot_features_0,tot_features_1)
        self.bn1 = T.nn.BatchNorm1d(tot_features_1)
        
        # risk section
        self.fc_risk_1      = T.nn.Linear(tot_features_1,tot_feaures_risk_1)
        self.bn_risk_1      = T.nn.BatchNorm1d(tot_feaures_risk_1)
        self.fc_risk_2      = T.nn.Linear(tot_feaures_risk_1,tot_feaures_risk_2)
        self.bn_risk_2      = T.nn.BatchNorm1d(tot_feaures_risk_2)
        self.fc_risk_final  = T.nn.Linear(tot_feaures_risk_2,tot_feaures_final)
        self.bn_risk_final  = T.nn.BatchNorm1d(tot_feaures_final)
        
    def forward(self, probs_softmax, encoding, residual, verbose = False):
        
        # for 224p images
        # conv forward for the 
        residual_out = self.e1(residual)
        residual_out = self.e2(residual_out)
        residual_out = self.e3(residual_out)
        residual_out = self.e4(residual_out)
        residual_out = self.e5(residual_out)    # 64, 7, 7 shape
        
        # print(residual_out.shape)
        
        # flat the residual
        flatten_residual = T.flatten(residual_out, start_dim=1)
        
        # print(flatten_residual.shape)
        
        # # build the vector input 
        x = T.cat((probs_softmax, encoding, flatten_residual), dim = 1)
        if verbose: print("input module b shape -> ", x.shape)
        
        # # preliminary layers
        x = self.gelu(self.bn1(self.fc1(x)))
        
        # # risk section
        x = self.gelu(self.bn_risk_1(self.fc_risk_1(x)))
        x = self.gelu(self.bn_risk_2(self.fc_risk_2(x)))
        x = self.gelu(self.bn_risk_final(self.fc_risk_final(x)))
        
        return x

#_____________________________________Test models_________________________________________________ 

class FC_classifier(nn.Module):
    """ 
    A simple ANN with fully connected layer + batch normalziation
    """
    def __init__(self, n_channel = 1, width = 28, height = 28, n_classes = 10):   # Default value for MNISt
        super(FC_classifier, self).__init__()
        
        fm          = 256 # feature map
        epsilon     = 0.001
        momentum    = 0.99

        self.flatten = nn.Flatten()  #input layer, flattening the image
        
        self.batch_norm1 = nn.BatchNorm1d(width * height * n_channel, eps= epsilon, momentum=momentum)
        self.fc1 = nn.Linear(width * height * n_channel, fm)

        self.batch_norm2 = nn.BatchNorm1d(fm, eps= epsilon, momentum=momentum)
        self.fc2 = nn.Linear(fm, fm)

        self.batch_norm3 = nn.BatchNorm1d(fm, eps= epsilon, momentum=momentum)
        self.fc3 = nn.Linear(fm, fm)

        self.batch_norm4 = nn.BatchNorm1d(fm, eps= epsilon, momentum=momentum)
        self.fc4 = nn.Linear(fm, n_classes)
        
        # activation functions
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        
        x = self.batch_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.batch_norm2(x)
        x = self.fc2(x)
        x = self.relu(x)

        x = self.batch_norm3(x)
        x = self.fc3(x)
        x = self.relu(x)

        x = self.batch_norm4(x)
        x = self.fc4(x)
        # x = self.softmax(x)     # don't use softmax, return the logits

        return x
    
    def getSummary(self, input_shape = (1,28,28), verbose = True):  #shape: color,width,height
        """
            input_shape -> tuple with simulated dimension used for the model summary
            expected input of this type -> color,width,height
        """
        if verbose: v = 1
        else: v = 0
        model_stats = summary(self, input_shape, verbose = v)
        
        return str(model_stats)
     
    def getLayers(self):
        return dict(self.named_parameters())
    
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
    
    def isCuda(self):
        return next(self.parameters()).is_cuda
         
def get_fc_classifier_Keras(input_shape = (28,28)):
    """ same as FC classifier implemented using keras moduel from tensorflows""" 
    
    # import_tf
    import tensorflow as tf
    from tensorflow import keras
    
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.relu),

    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.relu),

    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation=tf.nn.relu),

    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    return model

class TestAbnormModel(Project_abnorm_model):
    
    def __init__(self, shape_residual):
        
        # self.encoding_components = 500
        # self.residual_components = 5000
        # self.shape_softmax_probs = shape_softmax_probs
        # self.shape_encoding      = shape_encoding
        # self.shape_residual      = shape_residual
         
        # super().__init__((-1,*shape_softmax_probs[1:]), (-1,*shape_encoding[1:]), (-1,*shape_residual[1:]))
        super().__init__()
        self.residual_shape = shape_residual
        self.input_shape = shape_residual   # input_shape doesn't consider the batch
        self._createNet()
        self._init_weights_normal()
    
    def _createNet(self):
        # residual_length  = self.encoding_shape[1]  # flatten residual should have same dim of the encoding
        residual_length = 12544
        # tot_features_0          = self.shape_softmax_probs[1] + self.shape_encoding[1] + self.shape_residual[1]
        # tot_features_0          = self.probs_softmax_shape[1] +  self.encoding_shape[1] + residual_length


        # conv section
        self.e1 = Encoder_block(in_c =  self.residual_shape[1] , out_c = 16)
        self.e2 = Encoder_block(in_c =  16, out_c = 32)
        self.e3 = Encoder_block(in_c =  32, out_c = 64)
        # e2 = Encoder_block(in_c =  64, out_c = 128)
        


        tot_features_1      = 2048
        tot_features_2      = 1024
        
        
        # # taken from official work 
        tot_feaures_risk_1  = 512
        tot_feaures_risk_2  = 128
        tot_feaures_final   = 1
        
        self.gelu = T.nn.GELU()
        self.sigmoid = T.nn.Sigmoid()
        
        # preliminary layers
        self.fc1 = T.nn.Linear(residual_length,tot_features_1)
        self.bn1 = T.nn.BatchNorm1d(tot_features_1)
        self.fc2 = T.nn.Linear(tot_features_1,tot_features_2)
        self.bn2 = T.nn.BatchNorm1d(tot_features_2)
        
        # risk section
        self.fc_risk_1      = T.nn.Linear(tot_features_2,tot_feaures_risk_1)
        self.bn_risk_1      = T.nn.BatchNorm1d(tot_feaures_risk_1)
        self.fc_risk_2      = T.nn.Linear(tot_feaures_risk_1,tot_feaures_risk_2)
        self.bn_risk_2      = T.nn.BatchNorm1d(tot_feaures_risk_2)
        self.fc_risk_final  = T.nn.Linear(tot_feaures_risk_2,tot_feaures_final)
        self.bn_risk_final  = T.nn.BatchNorm1d(tot_feaures_final)
        
    def forward(self, residual, verbose = False):
        
        
        # conv forward for the 
        _, residual_out = self.e1(residual)
        _, residual_out = self.e2(residual_out)
        _, residual_out = self.e3(residual_out)
        
        # flat the residual
        flatten_residual = T.flatten(residual_out, start_dim=1)
        
        # build the vector input 
        # x = T.cat((probs_softmax, encoding, flatten_residual), dim = 1)
        # if verbose: print("input module b shape -> ", x.shape)
        
        # preliminary layers
        x = self.gelu(self.bn1(self.fc1(flatten_residual)))
        x = self.gelu(self.bn2(self.fc2(x)))
        
        # risk section
        x = self.gelu(self.bn_risk_1(self.fc_risk_1(x)))
        x = self.gelu(self.bn_risk_2(self.fc_risk_2(x)))
        x = self.gelu(self.bn_risk_final(self.fc_risk_final(x)))
        
        return x

if __name__ == "__main__":
    #                           [Start test section] 
    
    # setUp test
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    input_example = T.rand(size=(INPUT_CHANNELS,INPUT_HEIGHT,INPUT_WIDTH))
    
    # deepfake detection
    
    def test_ResNet():
        resnet = ResNet_scratch()
        resnet.to(device)
        print(resnet.isCuda())
        
       
        batch_example = input_example.unsqueeze(0)
        # print(batch_example.shape)
        resnet.getSummary(input_shape= input_example.shape)
        
    def test_ResNet50ImageNet():
        resnet = ResNet_ImageNet()
        resnet.to(device)
        resnet.getSummary(input_shape= input_example.shape)
        
    def test_simpleClassifier():
        classifier = FC_classifier(n_channel= 3, width = 256, height= 256)
        classifier.to(device)
        classifier.getSummary(input_shape= input_example.shape)
        
    def test_ResNet_Encoder_Decoder():
        
        model = ResNet_EDS(n_classes=2, use_upsample= False)
        model.to(device)
        print("device, encoder -> {}, decoder -> {}, scorer -> {}".format(model.getDevice(name_module="encoder"), model.getDevice(name_module="decoder"), model.getDevice(name_module="scorer")))
        # print(model.getLayers(name_module=None))
        model.getSummary()
        # model.getSummaryEncoder(input_shape=input_resnet_example.shape)        
        # model.getSummaryScorerPipeline(input_shape=input_resnet_example.shape)
        # model.getSummaryDecoderPipeline(input_shape = input_resnet_example.shape)
        # input_shape = (2048,1,1)
        # convTranspose2d_shapes(input_shape=input_shape, n_filters=128, kernel_size=5, padding=0, stride = 1, output_padding=2)

    def test_Unet():
        unet = Unet6()
        unet.to_device(device)
        x = T.rand((32, 3, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        print(x.shape)
        r,e = unet.forward(x)
        print(r.shape, e.shape)
        input("press enter to exit ")

    def test_UnetScorer():
        unet = Unet4_Scorer(n_classes=2, large_encoding=True)
        unet.to_device(device)
        print(unet.bottleneck_size)
        # unet.getSummary()
        
        x = T.rand((32, 3, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        # print(x.shape)
        logits, rec, enc = unet.forward(x)
        print(enc.shape)
        input("press enter to exit ")
    
    def test_UnetResidualScorer():
        # test residual conv block
        # enc_block = encoder_block_residual(128, 64)
        # 
        # print(x.shape)
        # y, p = enc_block.forward(x)
        # print(p.shape)
        
        x = T.rand((32, 3, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        # unet = Unet6L_ResidualScorer(n_channels=3, n_classes=2)
        unet = Unet6L_ResidualScorer(n_classes=2, large_encoding=True)
        unet.to_device(device)
        print(unet.bottleneck_size)
        # print(unet.bottleneck_size)
        # unet.getSummary()
        try:
            logits, rec, enc = unet.forward(x)
            print("logits shape: ", logits.shape)
        except:
            rec, enc = unet.forward(x)
            
        print("rec shape: ", rec.shape)
        print("enc shape: ", enc.shape)
        # input("press enter to exit ")
        
    def test_UnetScorerConfidence():
        unet = Unet4_Scorer_Confidence(n_classes=2, large_encoding=True)
        unet.to_device(device)
        print(unet.bottleneck_size)
        # unet.getSummary()
        
        x = T.rand((32, 3, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        # print(x.shape)
        logits, rec, enc, conf = unet.forward(x)
        print(logits.shape)
        print(rec.shape)
        print(enc.shape)
        print(conf.shape)
        # input("press enter to exit ")
    
    def test_VIT():
        
        # define test input
        x = T.rand((32, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        
        tests = [1]

        if tests[0]:
            # vit = ViT_base(n_classes=2)
            # vit = ViT(n_classes=2)
            # vit = ViT_b16_ImageNet().to(device=device)
            vit = ViT_timm().to(device = device)
            vit.getSummary()
            out = vit.forward(x)
            print(out.shape)
        
            input("press enter to exit ")
        
    def test_ViTEA():
        # define test input
        # x = T.rand((32, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)).to(device)

        img_1 = trans_input_base()(Image.open("./static/test_image_attention.png"))
        img_2 = trans_input_base()(Image.open("./static/test_image_attention2.png"))
        img_2 = img_2[:-1]
        
        print(img_1.shape)
        print(img_2.shape)
        

        x = T.stack([img_1, img_2], dim=0).cuda()
        
        # showImage(img_1)
        # showImage(img_2)
        
        print(x.shape)
        
        tests = [1, 0, 0]  # vit_EA, encoder, vit_EA + encoder
        
        if tests[0]:
            vit = ViT_timm_EA(use_attnmap_cls=True, resize_att_map=True, prog_model=3).to(device = device)
            # vit.getSummary()
            # print(vit.getAttributes())
            
        
            logits, encoding, attention = vit.forward(x)
            print("logits shape     -> ", logits.shape)
            print("encoding shape   -> ", encoding.shape)
            print("attention shape  -> ", attention.shape)

            print(T.sum(attention[0]))
            print(T.sum(attention[1]))
            
            # vit.getSummary()
            # print(vit.transform)
            # print(attention[0].shape)
            
            # att_map = attention[0]
            
            showImage(x[0])
            showImage(attention[0], has_color= False)
            
            blend, att = include_attention(x[0], attention[0])
            
            print(blend.shape)
            print(att.shape)
            
            showImage(blend)
            showImage(att)
            

        elif tests[1]:
            ae = AutoEncoder()
            ae.getSummary()
            
        elif tests[2]:
            vit = ViT_timm_EA().to(device = device)
            ae  = AutoEncoder().to(device=device)
            # vit.getSummary()
            # print(vit.getAttributes())
            logits, encoding, attention = vit.forward(x)
            # print(out[0])
            print(attention.shape)
            
            rec = ae(attention)
            print(rec.shape)
            
            showImage(attention[0], has_color= False)
            showImage(rec[0], has_color= False)
        
        input("press enter to exit ")
    
    # OOD detection
    
    def test_abnorm_basic():
        from    bin_classifier                  import DFD_BinClassifier_v4
        classifier = DFD_BinClassifier_v4(scenario="content", model_type="Unet4_Scorer")
        classifier.load("faces_Unet4_Scorer112p_v4_03-12-2023", 73)
        x_module_a = T.rand((32,INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        logits, reconstruction, encoding = classifier.model.forward(x_module_a)
        # input("press enter for next step ")
        
        softmax_prob = T.nn.functional.softmax(logits, dim=1)
        print("logits shape -> ", softmax_prob.shape)
        print("encoding shape -> ",encoding.shape)
        
        # from reconstuction to residual
        residual = T.square(reconstruction - x_module_a)
        
        
        print("residual shape ->", reconstruction.shape)

        abnorm_module = Abnormality_module_Basic(shape_softmax_probs = softmax_prob.shape, shape_encoding = encoding.shape, shape_residual = residual.shape).to(device)
        # abnorm_module.getSummary()
        y = abnorm_module.forward(logits, encoding, residual)
        print(y.shape)
        input("press enter to exit ")
    
    def test_abnorm_encoder():
        from    bin_classifier                  import DFD_BinClassifier_v4
        classifier = DFD_BinClassifier_v4(scenario="content", model_type="Unet4_Scorer")
        classifier.load("faces_Unet4_Scorer112p_v4_03-12-2023", 73)
        x_module_a = T.rand((32, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        logits, reconstruction, encoding = classifier.model.forward(x_module_a)
        # input("press enter for next step ")
        
        softmax_prob = T.nn.functional.softmax(logits, dim=1)
        print("logits shape -> ", softmax_prob.shape)
        print("encoding shape -> ",encoding.shape)
        
        # from reconstuction to residual
        residual = T.square(reconstruction - x_module_a)
        # residual_flatten = T.flatten(residual, start_dim=1)
        
        
        print("residual shape ->", residual.shape)
        # print("residual (flatten) shape ->",residual_flatten.shape)
        
        # abnorm_module = Abnormality_module_Encoder_v2(shape_softmax_probs = softmax_prob.shape, shape_encoding=encoding.shape, shape_residual=residual.shape).to(device)
        # abnorm_module.getSummary()
        # abnorm_module.forward(probs_softmax=softmax_prob, residual=residual, encoding=encoding)
        
        # abnorm_module = Abnormality_module_Encoder_v3(shape_softmax_probs = softmax_prob.shape, shape_encoding=encoding.shape, shape_residual=residual.shape).to(device)
        # abnorm_module.getSummary()
        # abnorm_module.forward(probs_softmax=softmax_prob, residual=residual, encoding=encoding)
        
        abnorm_module = Abnormality_module_Encoder_v3(shape_softmax_probs = softmax_prob.shape, shape_encoding=encoding.shape, shape_residual=residual.shape).to(device)
        # abnorm_module.getSummary()
        abnorm_module.forward(probs_softmax=softmax_prob, residual=residual, encoding=encoding)
        # input("press enter to exit ")
    
    # test using ViT trasformer
    def test_abnorm_encoder_vit():
        from    bin_ViTClassifier                  import DFD_BinViTClassifier_v7
        classifier = DFD_BinViTClassifier_v7(scenario="content")
        classifier.load("faces_ViTEA_timm_v7_07-02-2024", 21)
        x_module_a = T.rand((32, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        logits, encoding, att_map,  = classifier.model.forward(x_module_a)
        # input("press enter for next step ")
        
        softmax_prob = T.nn.functional.softmax(logits, dim=1)
        print("logits shape -> ", softmax_prob.shape)
        print("encoding shape -> ",encoding.shape)
        
        rec_att_map = classifier.autoencoder.forward(att_map)
        
        # from reconstuction to residual
        residual = T.square(att_map - rec_att_map)
        # residual_flatten = T.flatten(residual, start_dim=1)
        
        
        print("residual shape ->", residual.shape)
        # print("residual (flatten) shape ->",residual_flatten.shape)
        
        # test_encoding = T.rand((32, 50176)).to(device)
        # test_residual = T.rand((32, 3, 112, 112)).to(device)
        
        
        abnorm_module = Abnormality_module_Encoder_VIT_v3(shape_softmax_probs = softmax_prob.shape, shape_encoding=encoding.shape, shape_residual=residual.shape).to(device)
        # abnorm_module.getSummary()
        out = abnorm_module.forward(probs_softmax=softmax_prob, residual=residual, encoding=encoding)
        print(out.shape)
    
    # get_vitTimm_models()
    test_ViTEA()
    
    
    #                           [End test section] 
    