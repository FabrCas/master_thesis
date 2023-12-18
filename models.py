import time

import  torch                           as T
import  torch.nn.functional             as F
import  numpy                           as np
import  math
import  torch.nn                        as nn
from    torchsummary                    import summary
from    torchvision                     import models
from    torchvision.models              import ResNet50_Weights
from    utilities                       import print_dict, print_list, expand_encoding, convTranspose2d_shapes, get_inputConfig

# input settigs:
config = get_inputConfig()

INPUT_WIDTH     = config['width']
INPUT_HEIGHT    = config['height']
INPUT_CHANNELS  = config['channels']
UNET_EXP_FMS    = 4    # U-net power of 2 exponent for feature maps


# 1st models superclass
class Project_conv_model(nn.Module):
    
    def __init__(self, c,h,w, n_classes):
        super(Project_conv_model,self).__init__()
        
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
            summ = ""
            n_params = 0
            for k,v in self.getLayers().items():
                summ += "{:<30} -> {:<30}".format(k,str(tuple(v.shape))) + "\n"
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
        # Initialize the weights  using He initialization
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
    
            
    def forward(self):
        raise NotImplementedError

#_________________________________________ ResNet__________________________________________________

"""
                            ResNet (scratch) Legend:
                                
    c_n -> 2D/3D convolutional layer number n
    bn_n -> 2D/3D batch normalization layer number n
    relu -> rectified linear unit activation function
    mp -> 2D/3D max pooling layer
    ds_layer -> downsampling layer
    ap -> average pooling layer
    fm_dim -> feature map dimension
    do -> dropout layer
    exp_coeff -> expansion coefficent
    
    
    depth_level:
    0 -> 18  layers ResNet + 1 dropout
    1 -> 34  layers ResNet + 1 dropout
    2 -> 50  layers ResNet + 1 dropout
    3 -> 101 layers ResNet + 1 dropout
    4 -> 152 layers ResNet + 1 dropout
 
    
    input shape:
    2D -> [batch, colours, width, height]
"""
class Bottleneck_block2D_l(nn.Module):
    def __init__(self, n_inCh, n_outCh, stride = 1, ds_layer = None, exp_coeff = 4):
        super(Bottleneck_block2D_l,self).__init__()
        """
                            3 Convolutional layers
        """
        self.exp_coeff = exp_coeff

        # 1st block
        self.c_1 = nn.Conv2d(n_inCh, n_outCh, kernel_size=1, stride=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(n_outCh)
        
        # 2nd block
        self.c_2 = nn.Conv2d(n_outCh, n_outCh, kernel_size=3, stride=stride, padding=1)
        self.bn_2 = nn.BatchNorm2d(n_outCh)
        
        # 3rd block
        self.c_3 = nn.Conv2d(n_outCh, n_outCh*self.exp_coeff, kernel_size=1, stride=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(n_outCh*self.exp_coeff)
        
        # relu as a.f. for each block 
        self.relu = nn.ReLU()
        
        self.ds_layer  = ds_layer
        self.stride = stride
        
    
    def forward(self, x):
        x_init = x.clone()  # identity shortcuts
        
        # forwarding into the bottleneck layer
        
        x = self.relu(self.bn_1(self.c_1(x)))
        x = self.relu(self.bn_2(self.c_2(x)))
        x = self.bn_3(self.c_3(x))
    
        #downsample identity whether necessary and sum 
        if self.ds_layer is not None:
            x_init = self.ds_layer(x_init)
        
        x+=x_init
        x=self.relu(x)
        
        return x

class Bottleneck_block2D_s(nn.Module):
    
    def __init__(self, n_inCh, n_outCh, stride = 1, ds_layer = None, exp_coeff = 4):
        """
                            2 Convolutional layers
        """
        super(Bottleneck_block2D_s,self).__init__()
        self.exp_coeff = exp_coeff

        # 1st block
        self.c_1 = nn.Conv2d(n_inCh, n_outCh, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(n_outCh)
        
        # 2nd block
        self.c_2 = nn.Conv2d(n_outCh, n_outCh, kernel_size=3, stride = stride, padding=1)
        self.bn_2 = nn.BatchNorm2d(n_outCh)
        
        # relu as a.f. for each block 
        self.relu = nn.ReLU()
        
        self.ds_layer  = ds_layer
        self.stride = stride
        
    def forward(self, x):
        x_init = x.clone()  # identity shortcuts
        
        # forwarding into the bottleneck layer
        
        x = self.relu(self.bn_1(self.c_1(x)))
        x = self.bn_2(self.c_2(x))
        
        #downsample identity whether necessary and sum 
        if self.ds_layer is not None:
            x_init = self.ds_layer(x_init)
            
        x+=x_init
        x=self.relu(x)
        
        return x
      
class ResNet(Project_conv_model):
    # channel -> colors image
    # classes -> unique labels for the classification
    
    def __init__(self, depth_level = 2, n_classes = 10):
        super(ResNet,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        
        self.exp_coeff = 4 # output/input feature dimension ratio in bottleneck (default value for mid/big size model, reduced for small resnet 18 & 34)
    
        
        print("Initializing {} ...".format(self.__class__.__name__))
        self.initialTime = time.time()
        self.input_ch= 64   #    300 frames 64
        self.depth_level = depth_level
        self._check_depth_level()
        
        
        self.bottleneck_structs = [[2,2,2,2], [3,4,6,3], [3,4,6,3], [3,4,23,3], [3,8,36,3]]  # number of layers for the bottleneck                                             
        
        self.bottleneck_struct = self.bottleneck_structs[depth_level]
        self.fm_dim = [64,128,256,512]                          # feature map dimension
        
        
        self._create_net()
        
    
    def _check_depth_level(self):
        if self.depth_level< 0 or self.depth_level>4:
            raise ValueError("Wrong selection for the depth level, range: [0,4]")
        
        # set to 1 the exp coefficient if you are using reduced model
        if self.depth_level < 2: self.exp_coeff = 1
    
    
    def _create_net(self):
        # first block
        self.c_1 = nn.Conv2d(self.n_channels, self.fm_dim[0], kernel_size= 7, stride = 2, padding = 3, bias = False)
        self.bn_1 = nn.BatchNorm2d(self.fm_dim[0])
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size= 3, stride = 2, padding= 1)
        
        # body blocks
        self.l1 = self._buildLayers(n_blocks = self.bottleneck_struct[0], n_fm = self.fm_dim[0])
        self.l2 = self._buildLayers(n_blocks = self.bottleneck_struct[1], n_fm = self.fm_dim[1], stride = 2)
        self.l3 = self._buildLayers(n_blocks = self.bottleneck_struct[2], n_fm = self.fm_dim[2], stride = 2)
        self.l4 = self._buildLayers(n_blocks = self.bottleneck_struct[3], n_fm = self.fm_dim[3], stride = 2)
        
        
        # last block
        self.ap = nn.AdaptiveAvgPool2d((1,1)) # (1,1) output dimension
        self.do = nn.Dropout(0.3)
        self.fc = nn.Linear(512*self.exp_coeff, self.n_classes)
        
        # self.af_out = nn.Sigmoid() # used directly in the loss function
        
        print("Model created, time {} [s]".format(time.time() - self.initialTime))
        
    def _buildLayers(self, n_blocks, n_fm, stride = 1):
        
        """
            basic block 2 operations: conv + batch normalization
            bottleneck block 3 operations: conv + batch normalization + ReLU
        """
        list_layers = []
        
        # adapt x to be summed with output of the blocks (downsampling)
        if stride != 1 or self.input_ch != n_fm*self.exp_coeff: # so downsampling to handle the different shape of x 
            ds_layer = nn.Sequential(
                nn.Conv2d(self.input_ch, n_fm*self.exp_coeff, kernel_size=1, stride=stride),
                nn.BatchNorm2d(n_fm*self.exp_coeff)
                )
        else:
            ds_layer = None
    

        print("number of blocks fm {} -> {}".format(n_fm, n_blocks))
        # first layer to get the right feature map
        if self.depth_level < 2:
            print("building small block n°1")
            list_layers.append(Bottleneck_block2D_s(self.input_ch, n_fm, ds_layer= ds_layer, stride = stride))
        else:
            print("building big blocks n°1")
            list_layers.append(Bottleneck_block2D_l(self.input_ch, n_fm, ds_layer= ds_layer, stride = stride))
        self.input_ch = n_fm * self.exp_coeff
        
        # include all the layers from the bottleneck blocks
        for index, _ in enumerate(range(n_blocks -1)):
            if self.depth_level < 2:
                print("building small block n°{}".format(index +2))
                list_layers.append(Bottleneck_block2D_s(self.input_ch, n_fm))
            else:
                print("building big blocks n°{}".format(index +2))
                list_layers.append(Bottleneck_block2D_l(self.input_ch, n_fm))
        
        return nn.Sequential(*list_layers)
        
    
    def forward(self, x):
        
        # first block
        x = self.mp(self.relu(self.bn_1(self.c_1(x))))
        
        # body blocks
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
    
        # last block
        x = self.ap(x)
        x = x.reshape(x.shape[0], -1)  #(batch_size, all_values)
        x = self.do(x)
        x = self.fc(x)
        
        return x
    
#_____________________________________ResNet 50 ImageNet___________________________________________

class ResNet_ImageNet(Project_conv_model):   # not nn.Module subclass, but still implement forward method calling the one of the model
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
    
# _____________________________________ U net ______________________________________________________

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

class Unet4(Project_conv_model):
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

class Unet5(Project_conv_model):
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

class Unet6(Project_conv_model):
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

    
#_____________________________________ OOD custom modules __________________________________________

#                                       custom ResNet

class ResNet_EDS(Project_conv_model): 
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
        
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)
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
class Unet4_Scorer(Project_conv_model):
    """
        U-net 4 + Scorer, 4 encoders and 4 decoders
    """

    def __init__(self, n_classes= 10):
        super(Unet4_Scorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        # self.bottleneck_size = int(self.feature_maps(4)*(self.width/16)*(self.width/16))
        self.bottleneck_size = int(self.feature_maps(4)*math.floor(self.width/16)**2) 
        self.n_levels = 4
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


    def forward(self, x):
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
        # print(enc.shape)
        
        # classification
        # f1          = self.relu(self.bn1(self.fc1(enc)))
        # f1_drop     = self.do(f1)
        # f2          = self.relu(self.bn2(self.fc2(f1_drop)))
        # f2_drop     = self.do(f2)
        # logits      = self.fc3(f2_drop)
        
        out         = self.relu(self.bn1(self.fc1(enc)))
        out         = self.do(out)
        out         = self.relu(self.bn2(self.fc2(out)))
        out         = self.do(out)
        out         = self.relu(self.bn3(self.fc3(out)))
        out         = self.do(out)
        out         = self.relu(self.bn4(self.fc4(out)))
        out         = self.do(out)
        logits      = self.fc5(out)
        
        
        

        # decoder 
        d1 = self.d1(bottleneck, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d4))  # check sigmoid vs tanh
        return logits, rec, enc

class Unet5_Scorer(Project_conv_model):
    """
        U-net 5 + Scorer, 5 encoders and 5 decoders
    """

    def __init__(self, n_classes= 10):
        super(Unet5_Scorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.bottleneck_size = int(self.feature_maps(5)*math.floor(self.width/32)**2)  
        self.n_levels = 5     
        
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

    def forward(self, x):
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
        
        # classification
        # f1          = self.relu(self.bn1(self.fc1(enc)))
        # f1_drop     = self.do(f1)
        # f2          = self.relu(self.bn2(self.fc2(f1_drop)))
        # f2_drop     = self.do(f2)
        # logits      = self.fc3(f2_drop)
        out         = self.relu(self.bn1(self.fc1(enc)))
        out         = self.do(out)
        out         = self.relu(self.bn2(self.fc2(out)))
        out         = self.do(out)
        out         = self.relu(self.bn3(self.fc3(out)))
        out         = self.do(out)
        out         = self.relu(self.bn4(self.fc4(out)))
        out         = self.do(out)
        logits      = self.fc5(out)
        

        # decoder 
        d1 = self.d1(bottleneck, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d5))  # check sigmoid vs tanh
        return logits, rec, enc

class Unet6_Scorer(Project_conv_model):
    """
        U-net 6 + Scorer, 6 encoders and 6 decoders.
        This version include an additional layer for the scorer
    """

    def __init__(self, n_classes= 10):
        super(Unet6_Scorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.n_levels = 6
        
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


    def forward(self, x):
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
        
        
        # classification
        out         = self.relu(self.bn1(self.fc1(enc)))
        out         = self.do(out)
        out         = self.relu(self.bn2(self.fc2(out)))
        out         = self.do(out)
        out         = self.relu(self.bn3(self.fc3(out)))
        out         = self.do(out)
        out         = self.relu(self.bn4(self.fc4(out)))
        out         = self.do(out)
        logits      = self.fc5(out)

        
        d1 = self.d1(bottleneck, s6)
        d2 = self.d2(d1, s5)
        d3 = self.d3(d2, s4)
        d4 = self.d4(d3, s3)
        d5 = self.d5(d4, s2)
        d6 = self.d6(d5, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d6))  # check sigmoid vs tanh
        return logits, rec, enc

class Unet6L_Scorer(Project_conv_model):
    """
        U-net 6 + Scorer, 6 encoders and 6 decoders.
        This version include an additional layer for the scorer and LargeConv_block instead of Conv_blocks
    """

    def __init__(self, n_classes= 10):
        super(Unet6L_Scorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.bottleneck_size = int(self.feature_maps(6)*math.floor(self.width/64)**2)
        self.n_levels = 6
        
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

    
    def forward(self, x):
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
        
        # print(bottleneck.shape)
        # print(enc.shape)
        # print(self.bottleneck_size)
        
        # classification
        out         = self.relu(self.bn1(self.fc1(enc)))
        out         = self.do(out)
        out         = self.relu(self.bn2(self.fc2(out)))
        out         = self.do(out)
        out         = self.relu(self.bn3(self.fc3(out)))
        out         = self.do(out)
        out         = self.relu(self.bn4(self.fc4(out)))
        out         = self.do(out)
        logits      = self.fc5(out)
        
        d1 = self.d1(bottleneck, s6)
        d2 = self.d2(d1, s5)
        d3 = self.d3(d2, s4)
        d4 = self.d4(d3, s3)
        d5 = self.d5(d4, s2)
        d6 = self.d6(d5, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d6))  # check sigmoid vs tanh
        return logits, rec, enc

class Unet4_ResidualScorer(Project_conv_model):
    """
        U-net 4 with Resiudal Encoder + Scorer, 5 encoders and 5 decoders
    """

    def __init__(self, n_classes= 10):
        super(Unet4_ResidualScorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS     # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        # self.bottleneck_size = int(self.feature_maps(4)*(self.width/8)*(self.width/8))  (224x224 case)
        self.bottleneck_size = int(self.feature_maps(4)*math.floor(self.width/16)**2) 
        self.residual2conv = False  # if False identity shortcuts are connected after the pooling layer
        self.n_levels = 4
        
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

    def forward(self, x):
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
        
        # classification
        # f1          = self.relu(self.bn1(self.fc1(enc)))
        # f1_drop     = self.do(f1)
        # f2          = self.relu(self.bn2(self.fc2(f1_drop)))
        # f2_drop     = self.do(f2)
        # logits      = self.fc3(f2_drop)
        
        out         = self.relu(self.bn1(self.fc1(enc)))
        out         = self.do(out)
        out         = self.relu(self.bn2(self.fc2(out)))
        out         = self.do(out)
        out         = self.relu(self.bn3(self.fc3(out)))
        out         = self.do(out)
        out         = self.relu(self.bn4(self.fc4(out)))
        out         = self.do(out)
        logits      = self.fc5(out)
        
        # decoder 
        d1 = self.d1(bottleneck, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d4))  # check sigmoid vs tanh
        return logits, rec, enc

class Unet5_ResidualScorer(Project_conv_model):
    """
        U-net 5 with Resiudal Encoder + Scorer, 5 encoders (residual) and 5 decoders
    """
    
    def __init__(self, n_classes= 10):
        super(Unet5_ResidualScorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        self.bottleneck_size = int(self.feature_maps(5)*math.floor(self.width/32)**2) 
        self.residual2conv = False  # if False identity shortcuts are connected after the pooling layer
        self.n_levels = 5
        
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

    def forward(self, x):
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
        # print(enc.shape)
        
        # classification
        # f1          = self.relu(self.bn1(self.fc1(enc)))
        # f1_drop     = self.do(f1)
        # f2          = self.relu(self.bn2(self.fc2(f1_drop)))
        # f2_drop     = self.do(f2)
        # logits      = self.fc3(f2_drop)
        out         = self.relu(self.bn1(self.fc1(enc)))
        out         = self.do(out)
        out         = self.relu(self.bn2(self.fc2(out)))
        out         = self.do(out)
        out         = self.relu(self.bn3(self.fc3(out)))
        out         = self.do(out)
        out         = self.relu(self.bn4(self.fc4(out)))
        out         = self.do(out)
        logits      = self.fc5(out)
        
        

        # decoder 
        d1 = self.d1(bottleneck, s5)
        d2 = self.d2(d1, s4)
        d3 = self.d3(d2, s3)
        d4 = self.d4(d3, s2)
        d5 = self.d5(d4, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d5))  # check sigmoid vs tanh
        return logits, rec, enc
    
class Unet6_ResidualScorer(Project_conv_model):
    """
        U-net 6 + Scorer, 6 encoders (residual) and 6 decoders.
        This version include an additional layer for the scorer
    """

    def __init__(self, n_classes = 10):
        super(Unet6_ResidualScorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        # self.bottleneck_size = int(self.feature_maps(6)*(w/64)*(w/64))
        self.bottleneck_size = int(self.feature_maps(6)*math.floor(self.width/64)**2)  # if 224x224 the width is divided by 32
        self.residual2conv = False  # if False identity shortcuts are connected after the pooling layer
        self.n_levels = 6
        
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

    def forward(self, x):
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
        
        # classification
        out         = self.relu(self.bn1(self.fc1(enc)))
        out         = self.do(out)
        out         = self.relu(self.bn2(self.fc2(out)))
        out         = self.do(out)
        out         = self.relu(self.bn3(self.fc3(out)))
        out         = self.do(out)
        out         = self.relu(self.bn4(self.fc4(out)))
        out         = self.do(out)
        logits      = self.fc5(out)

        d1 = self.d1(bottleneck, s6)
        d2 = self.d2(d1, s5)
        d3 = self.d3(d2, s4)
        d4 = self.d4(d3, s3)
        d5 = self.d5(d4, s2)
        d6 = self.d6(d5, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d6))  # check sigmoid vs tanh
        return logits, rec, enc

class Unet6L_ResidualScorer(Project_conv_model):
    """
        U-net 6 + Scorer, 6 encoders (residual) and 6 decoders.
        This version include an additional layer for the scorer and LargeConv_block instead of Conv_blocks
    """

    def __init__(self, n_classes= 10):
        super(Unet6L_ResidualScorer,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        print("Initializing {} ...".format(self.__class__.__name__))

        self.features_order = UNET_EXP_FMS   # orders greater and equal than 5 saturates the GPU!
        self.feature_maps = lambda x: int(math.pow(2, self.features_order+x))  # x depth block in u-net
        # self.bottleneck_size = int(self.feature_maps(6)*(w/64)*(w/64))
        self.bottleneck_size = int(self.feature_maps(6)*math.floor(self.width/64)**2)  # if 224x224 the width is divided by 32
        self.residual2conv = False  # if False identity shortcuts are connected after the pooling layer
        self.n_levels = 6
        
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

    def forward(self, x):
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
        
        # classification
        out         = self.relu(self.bn1(self.fc1(enc)))
        out         = self.do(out)
        out         = self.relu(self.bn2(self.fc2(out)))
        out         = self.do(out)
        out         = self.relu(self.bn3(self.fc3(out)))
        out         = self.do(out)
        out         = self.relu(self.bn4(self.fc4(out)))
        out         = self.do(out)
        logits      = self.fc5(out)

        d1 = self.d1(bottleneck, s6)
        d2 = self.d2(d1, s5)
        d3 = self.d3(d2, s4)
        d4 = self.d4(d3, s3)
        d5 = self.d5(d4, s2)
        d6 = self.d6(d5, s1)
        
        # reconstuction
        rec = self.decoder_out_fn(self.out(d6))  # check sigmoid vs tanh
        return logits, rec, enc

#                                       custom abnormality module

# 2nd models superclass
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
    """ it reduces the spatial dimensionality of half """ 
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride = 2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride = 2, padding=1)
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


class Abnormality_module_Encoder_v2(Project_abnorm_model):
    
    """
    
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

class TestModel(Project_abnorm_model):
    
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

       
#_____________________________________Vision Transformer (ViT)_____________________________________        


#_____________________________________Other models_________________________________________________ 

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

if __name__ == "__main__":
    #                           [Start test section] 
    
    # setUp test
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    input_resnet_example = T.rand(size=(INPUT_CHANNELS,INPUT_HEIGHT,INPUT_WIDTH))
    
    def test_ResNet():
        resnet = ResNet()
        resnet.to(device)
        print(resnet.isCuda())
        
       
        batch_example = input_resnet_example.unsqueeze(0)
        # print(batch_example.shape)
        resnet.getSummary(input_shape= input_resnet_example.shape)
        
    def test_ResNet50ImageNet():
        resnet = ResNet_ImageNet()
        resnet.to(device)
        resnet.getSummary(input_shape= input_resnet_example.shape)
        
    def test_simpleClassifier():
        classifier = FC_classifier(n_channel= 3, width = 256, height= 256)
        classifier.to(device)
        classifier.getSummary(input_shape= input_resnet_example.shape)
        
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
        unet = Unet4_Scorer(n_classes=2)
        unet.to_device(device)
        print(unet.bottleneck_size)
        # unet.getSummary()
        
        x = T.rand((32, 3, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        # print(x.shape)
        logits, rec, enc = unet.forward(x)
        # input("press enter to exit ")
    
    def test_UnetResidualScorer():
        # test residual conv block
        # enc_block = encoder_block_residual(128, 64)
        # 
        # print(x.shape)
        # y, p = enc_block.forward(x)
        # print(p.shape)
        
        x = T.rand((32, 3, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
        # unet = Unet6L_ResidualScorer(n_channels=3, n_classes=2)
        unet = Unet6L_ResidualScorer(n_classes=2)
        unet.to_device(device)
        # print(unet.bottleneck_size)
        unet.getSummary()
        try:
            logits, rec, enc = unet.forward(x)
            print("logits shape: ", logits.shape)
        except:
            rec, enc = unet.forward(x)
            
        print("rec shape: ", rec.shape)
        print("enc shape: ", enc.shape)
        # input("press enter to exit ")
        
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
        
        # abnorm_module = Abnormality_module_Encoder(shape_softmax_probs = softmax_prob.shape, shape_encoding=encoding.shape, shape_residual=residual.shape).to(device)
        # abnorm_module.getSummary()
        
        model = TestModel(residual.shape)
        model.getSummary()
        
        # input("press enter to exit ")
    
    test_abnorm_encoder()
    #                           [End test section] 
    