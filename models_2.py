from models import *

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
      
class ResNet_scratch(Project_DFD_model):
    # channel -> colors image
    # classes -> unique labels for the classification
    
    def __init__(self, depth_level = 2, n_classes = 10):
        super(ResNet_scratch,self).__init__(c = INPUT_CHANNELS, h = INPUT_HEIGHT, w = INPUT_WIDTH, n_classes = n_classes)
        
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
    

# _____________________________ViT base: 1st implementation _______________________________________

class FCPatchEmbedding(nn.Module):
    def __init__(self, in_channels = INPUT_CHANNELS, patch_size = PATCH_SIZE, emb_size = EMB_SIZE):
        """ FC linear projection
        
        
        intended for squared patches of size in_channels x patch_size x patch_size
        
        """
        super().__init__()
        self.in_channels    = in_channels
        self.patch_size     = patch_size
        self.emb_size       = emb_size
        self.projection     = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_size * self.patch_size * self.in_channels, self.emb_size)
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        x = self.projection(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout_percentage):
        super(Attention, self).__init__()
        self.dim                = dim
        self.n_heads            = n_heads
        self.dropout_percentage = dropout_percentage
        self.multi_head         = T.nn.MultiheadAttention(
            embed_dim   =   self.dim,
            num_heads   =   self.n_heads,
            dropout     =   self.dropout_percentage,
            batch_first =   True)     # ??
        self.q                  = T.nn.Linear(self.dim, self.dim)
        self.k                  = T.nn.Linear(self.dim, self.dim)
        self.v                  = T.nn.Linear(self.dim, self.dim)
        
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        out, _ = self.multi_head(q,k,v)
        return out
        
class PreNorm(nn.Module):
    """ a simple wrapper class that applies Layer normalization before forwarding a specified function fn """     
    
    def __init__(self, dim, fn):
        super(PreNorm,self).__init__()
        self.norm   = nn.LayerNorm(dim)
        self.fn     = fn
    
    def forward(self,x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  

class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout_percentage = 0.0):
        super(FeedForward, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_percentage),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_percentage)     
        )

class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x 

class ViT_base(Project_DFD_model):
    
    def __init__(self, n_channels = INPUT_CHANNELS, img_width = INPUT_WIDTH, img_height = INPUT_HEIGHT, 
                 patch_size = PATCH_SIZE, emb_size = EMB_SIZE, n_layers = 4, n_classes = 10,
                 dropout = 0.1, n_heads = 2):
        
        super(ViT_base, self).__init__(c = n_channels,h = img_height,w = img_width, n_classes = n_classes)
        
        self.n_channels = n_channels
        self.width          = img_width
        self.height         = img_height
        self.img_size       = img_height   # supposing squred image take arbitrarily width or height
        self.input_shape    = (self.n_channels, self.height, self.width)
        self.patch_size     = patch_size
        self.emb_size       = emb_size
        self.n_layers       = n_layers 
        self.n_classes      = n_classes
        self.dropout        = dropout
        self.n_heads        = n_heads
        
        # compute the total number of patches for each image and the define the CLS token
        self.n_patches = (self.img_size//self.patch_size)**2
        self.cls_token = nn.Parameter(T.rand(1,1,self.emb_dim))
        
        # Embeddings
        self.patch_embedding    = FCPatchEmbedding(in_channels= self.n_channels, patch_size= self.patch_size, emb_size= self.emb_dim)
        self.pos_embedding      = nn.Parameter(T.rand(1, self.n_patches +1, self.emb_dim))                                             # +1 for [CLS] token
        
        # Transformer Encoder
        self.layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            transformer_block = nn.Sequential(
                ResidualBlock(PreNorm(self.emb_dim, Attention(dim = self.emb_dim, n_heads= self.n_heads, dropout_percentage= self.dropout))),
                ResidualBlock(PreNorm(self.emb_dim, FeedForward(dim = self.emb_dim, hidden_dim= self.emb_dim, dropout_percentage= self.dropout)))
            )
            self.layers.append(transformer_block)
            
        # Classification Head
        self.head = nn.Sequential(nn.LayerNorm(self.emb_dim), nn.Linear(self.emb_dim, self.n_classes))
        
        self._init_weights_normal()
    
        
    def forward(self, x):
        # patch embedding 
        x = self.patch_embedding(x)
        # get shape data
        b, n, _ = x.shape
        
        # include the CLS token to inputs, repeating for the number of batches
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = T.cat((cls_token, x), dim = 1)
        x += self.pos_embedding[:, :(n+1)]   # x += self.pos_embedding ? 
        
        # Forward thorugh Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)
           
        # classification head forward only for the cls token
        logits = self.head(x[:, 0, :])
        
        return logits


# _____________________________ViT base: 2nd implementation _______________________________________

class ConvPatchEmbedding(nn.Module):
    def __init__(self, img_size = INPUT_WIDTH, patch_size= PATCH_SIZE, in_channels= INPUT_CHANNELS, emb_size=768):
        super(ConvPatchEmbedding, self).__init__()
        self.patch_embed = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.img_size = img_size
        
    def forward(self, x):
        # print(x.shape)
        x = self.patch_embed(x)
        # print(x.shape)
        x = x.flatten(2).transpose(1, 2)   # transpose change the dimension between encoding and sequence of patches
        # print(x.shape)
        return x

class ViT_base_2(Project_DFD_model):
    def __init__(self, img_size = INPUT_WIDTH, patch_size= PATCH_SIZE, in_channels= INPUT_CHANNELS, emb_size=768, n_heads=12, n_layers=12, n_classes=10):
        super(ViT_base_2, self).__init__(c = in_channels,h = img_size,w = img_size, n_classes = n_classes)
        
        # 
        self.emb_size   = emb_size 
        self.patch_size = patch_size
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.patch_embedding = ConvPatchEmbedding(img_size, patch_size, in_channels, emb_size)
        self.num_patches = (img_size // patch_size) ** 2
        # self.pos_embedding = nn.Parameter(T.randn(1, self.num_patches + 1, emb_size))  # +1 for [CLS] token
        
        
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_size,
                nhead=n_heads,
                dim_feedforward=4 * emb_size,
                batch_first= True
            ),
            num_layers=n_layers,
        )
        self.head = nn.Linear(emb_size, n_classes)
        # self.cls_token = nn.Parameter(T.randn(1, 1, emb_size))
        
        self._init_weights_normal()
        
        self.pos_embedding = nn.Parameter(self._get_positional_encoding(emb_size, self.num_patches))


    def _get_positional_encoding(self, emb_size, num_patches):
        position = T.arange(0, num_patches).unsqueeze(1).float()
        div_term = T.exp(T.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size))
        pos_encoding = T.zeros(1, num_patches, emb_size)
        pos_encoding[:, :, 0::2] = T.sin(position * div_term)
        pos_encoding[:, :, 1::2] = T.cos(position * div_term)
        return pos_encoding


    def forward(self, x):
        x = self.patch_embedding(x)
        # cls_token = self.cls_token.expand(x.size(0), -1, -1)  # Expand cls_token to match batch size
        # x = T.cat((cls_token, x), dim=1)
        
        x = x + self.pos_embedding[:, :x.size(1)]  # Add positional encoding
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        # x = self.fc(x)
        # logits = self.head(x[:, 0, :])
        logits = self.head(x)

        return logits
