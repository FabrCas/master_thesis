import os
import sys

try:
    # Add the project root directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from models import * 
except:
    from models import * 
    
# deepfake detection
def test_ResNet50ImageNet():
    resnet = ResNet_ImageNet()
    resnet.to(device)
    resnet.getSummary(input_shape= input_example.shape)
    # batch_example.to(device)
    resnet.forward(batch_example.to(device))
    input("press enter to exit ")
    
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
    model.forward(batch_example.to(device))
    # model.getSummaryEncoder(input_shape=input_resnet_example.shape)        
    # model.getSummaryScorerPipeline(input_shape=input_resnet_example.shape)
    # model.getSummaryDecoderPipeline(input_shape = input_resnet_example.shape)
    # input_shape = (2048,1,1)
    # convTranspose2d_shapes(input_shape=input_shape, n_filters=128, kernel_size=5, padding=0, stride = 1, output_padding=2)
    input("press enter to exit ")
    
def test_Unet():
    unet = Unet6()
    unet.to_device(device)
    x = T.rand((32, 3, INPUT_HEIGHT, INPUT_WIDTH)).to(device)
    print(x.shape)
    r,e = unet.forward(x)
    print(r.shape, e.shape)
    input("press enter to exit ")

def test_UnetScorer():
    unet = Unet5_Scorer(n_classes=2, large_encoding=True)
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
    unet = Unet5_ResidualScorer(n_classes=2, large_encoding=True)
    unet.to_device(device)
    # print(unet.bottleneck_size)
    # print(unet.bottleneck_size)
    # unet.getSummary()
    try:
        logits, rec, enc = unet.forward(x)
        print("logits shape: ", logits.shape)
    except:
        rec, enc = unet.forward(x)
        
    # print("rec shape: ", rec.shape)
    # print("enc shape: ", enc.shape)
    input("press enter to exit ")
    
def test_UnetScorerConfidence():
    unet = Unet5_Scorer_Confidence(n_classes=2, large_encoding=True)
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
    input("press enter to exit ")

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
    
    tests = [0, 1, 0]  # vit_EA, encoder, vit_EA + encoder
    
    if tests[0]:
        vit = ViT_timm_EA(resize_att_map=True, prog_model=3).to(device = device)
        # vit.getSummary()
        # print(vit.getAttributes())
        
    
        out = vit.forward(x)
        logits          = out[0]
        encoding        = out[1]
        attention       = out[2]
        attention_full  = out[3]
        
        input("press something to continue")
        
        print("logits shape     -> ", logits.shape)
        print("encoding shape   -> ", encoding.shape)
        print("attention cls shape  -> ", attention.shape)
        print("attention full shape  -> ", attention_full.shape)

        print(T.sum(attention[0]))
        print(T.sum(attention[1]))
        
        # vit.getSummary()
        # print(vit.transform)
        # print(attention[0].shape)
        
        # att_map = attention[0]
        
        showImage(x[0])
        showImage(attention[0], has_color= False)
        showImage(attention_full[1], has_color= False)
        
        blend, att = include_attention(x[0], attention[0])
        
        print(blend.shape)
        print(att.shape)
        
        showImage(blend)
        showImage(att)
        

    elif tests[1]:
        # ae = AutoEncoder()
        ae = VAE()
        ae.getSummary()
        
        batch_example = T.rand(size=(32,1,INPUT_HEIGHT,INPUT_WIDTH)).to(device)
        ae.forward(batch_example)
        input("press something to continue")
        
    elif tests[2]:
        vit = ViT_timm_EA().to(device = device)
        ae  = AutoEncoder().to(device=device)
        # vit.getSummary()
        # print(vit.getAttributes())
        # logits, encoding, attention = vit.forward(x)
        
        out = vit.forward(x)
        logits      = out[0]
        encoding    = out[1]
        attention   = out[2]
        
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
    classifier = DFD_BinClassifier_v4(scenario="content", model_type="Unet5_Residual_Scorer", large_encoding=True)
    # classifier.load("faces_Unet4_Scorer112p_v4_03-12-2023", 73)
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
    
    abnorm_module = Abnormality_module_Encoder_v4(shape_softmax_probs = softmax_prob.shape, shape_encoding=encoding.shape, shape_residual=residual.shape).to(device)
    # abnorm_module.getSummary()
    abnorm_module.forward(probs_softmax=softmax_prob, residual=residual, encoding=encoding)
    input("press enter to exit ")

# test using ViT trasformer
def test_abnorm_encoder_vit():
    from    bin_ViTClassifier                  import DFD_BinViTClassifier_v7
    classifier = DFD_BinViTClassifier_v7(scenario="content")
    classifier.load("faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_13-02-2024", 25)
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
    
    
    abnorm_module = Abnormality_module_Encoder_ViT_v4(shape_softmax_probs = softmax_prob.shape, shape_encoding=encoding.shape, shape_residual=residual.shape).to(device)
    # abnorm_module.getSummary()
    out = abnorm_module.forward(probs_softmax=softmax_prob, residual=residual, encoding=encoding, verbose = True)
    print(out.shape)
    input("press enter to exit ")
    
if __name__ == "__main__":

    # setUp test
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    input_example = T.rand(size=(INPUT_CHANNELS,INPUT_HEIGHT,INPUT_WIDTH))
    batch_example = T.rand(size=(32,INPUT_CHANNELS,INPUT_HEIGHT,INPUT_WIDTH))