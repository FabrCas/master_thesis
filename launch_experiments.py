from experiments import * 


# 1) pytorch implementation
def train_baseline_T():            
    classifier = MNISTClassifier()
    classifier.train()
    classifier.load()

# tensorflow implementation
def train_baseline_tf():
    classifier = MNISTClassifier_keras()
    classifier.train()
    classifier.load_model()

# train decoders
def train_ResNetED_content():
    model = Decoder_ResNetEDS(scenario="content")
    model.train(name_train="faces_resnet50ED")

def train_Unet_content(): 
    model = Decoder_Unet(scenario="content")
    model.train(name_train="faces_Unet4")

# show results from end-dec
def showReconstructionResnetED(name_model, epoch, scenario):
    
    n_picture = 0
    model = Decoder_ResNetEDS(scenario = scenario, useGPU= True)
    model.load(name_model, epoch)
    img, _ = model.train_dataset.__getitem__(n_picture)
    showImage(img, name = "original_"+ str(n_picture), save_image = True)
    img     = T.unsqueeze(img, dim= 0).to(model.device)
    enc     = model.model.encoder_module.forward(img)
    rec_img = model.model.decoder_module.forward(enc)
    rec_img = T.squeeze(rec_img, dim = 0)
    showImage(rec_img, name="reconstructed_"+ str(n_picture) + "_decoding_" + name_model, save_image = True)
    
def showReconstructionUnet(name_model, epoch, scenario):
    n_picture = 3
    decoder = Decoder_Unet(scenario = scenario, useGPU= True)
    decoder.load(name_model, epoch)
    img, _ = decoder.train_dataset.__getitem__(n_picture)
    showImage(img, name = "original_"+ str(n_picture), save_image = True)
    img     = T.unsqueeze(img, dim= 0).to(decoder.device)
    
    rec_img, _ = decoder.model.forward(img)
    rec_img = T.squeeze(rec_img, dim = 0)
    
    showImage(rec_img, name="reconstructed_"+ str(n_picture) + "_decoding_" + name_model, save_image = True)

# test ViTEA
def train_ViT_Cifar():
    classifier = CIFAR_ViT_Classifier()
    # classifier.train()
    classifier.load()
    classifier.test_accuracy()

def train_VITEA_CIFAR():
    classifier = CIFAR_ViTEA_Classifier(prog_model = 3, cifar100=False)
    classifier.train()
    # classifier.load()
    classifier.test_accuracy()
    
def test_attention_map():
    classifier =  CIFAR_ViTEA_Classifier(prog_model = 3, cifar100=False)
    classifier.load()
    data_iter = classifier.train_data
    save    = False
    img_id  = 0
    
    img, y = data_iter.__getitem__(img_id)

        
    showImage(img, save_image= save, name="attention_original_" + str(img_id))

    
    img = img.unsqueeze(dim=0)
    print(img.shape)
    
    
    img = img.to(device = classifier.device)
    
    _, _, att_map = classifier.model.forward(img)
    
    print(att_map.shape, T.max(att_map), T.min(att_map))
    

    showImage(att_map[0], has_color= False, save_image= save, name="attention_map_" + str(img_id))

    # show_imgs_blend(img_d, att_map.cpu(), alpha=0.8, save_image= save, name="attention_blend_" + str(img_id))
    result, att_map  = include_attention(img, att_map, alpha= 0.7)
    
    print(att_map.shape, T.max(att_map), T.min(att_map))
    
    print(result.shape)
    
    showImage(result[0], save_image= save, name="attention_map_" + str(img_id))

#                           [Benchmark functions]

#                               train & test classifier
def train_classifier_benchmark(add2name, prog_model = 3, cifar100 = False, image_size = 224):
    
    vitea = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model=prog_model, image_size = image_size)
    vitea.train_classifier(add_in_name= add2name)
    # vitea.test(epoch=50, name_folder= "train_50_epochs_22-02-2024")

def continue_train_classifier_benchmark(name_folder, epoch_start, end_epoch, prog_model = 3, cifar100 = False, image_size = 224):
    vitea = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model=prog_model, image_size=image_size)
    vitea.load_classifier(name_folder=name_folder, epoch=epoch_start)
    vitea.train_classifier(start_epoch=epoch_start, end_epoch= end_epoch)
    # vitea
    
def train_AE_benchmark(name_folder, epoch_classifier, prog_model = 3, cifar100 = False, image_size = 224):
    vitea = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model= prog_model, image_size= image_size)
    vitea.load_classifier(name_folder=name_folder, epoch= epoch_classifier)
    vitea.train_ae(name_folder=name_folder)
    
def test_classifier_benchmark(name_folder, epoch_classifier, prog_model = 3, cifar100 = False, image_size = 224):
    vitea = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model= prog_model, image_size=image_size)
    vitea.test(name_folder= name_folder, epoch=epoch_classifier)   # load is peformed directly in the test function

#                               train & test abn module

def load_classifier_benchmark():
    cifar100            = False
    prog_model          = 3
    name_folder         = "train_3_DeiT_tiny_27-02-2024"
    epoch_classifier    = 50
    epoch_autoencodder  = 50
    classifier = CIFAR_VITEA_benchmark(cifar100=cifar100, prog_model= prog_model)
    classifier.load_classifier(epoch=epoch_classifier, name_folder=name_folder)
    classifier.load_autoencoder(epoch=epoch_autoencodder, name_folder= name_folder)
    print("\n\n\n###################### ID classifier loaded ####################\n\n\n")
    return classifier

def train_abn_module(model_type = "encoder_v3", image_size = 224, add2name = "", with_outlier = False):
    
    classifier = load_classifier_benchmark()
    abn = CIFAR_VITEA_Abnormality_module(classifier, model_type= model_type, image_size= image_size, with_outlier= with_outlier)
    if add2name != "": add2name = "_" + add2name 
    abn.train(additional_name= "test"+ add2name)
    
def test_abn_module(name_folder, ood_dataset, epoch, model_type= "encoder_v3", image_size = 224, with_outlier = False): 
    
    classifier = load_classifier_benchmark()
    abn = CIFAR_VITEA_Abnormality_module(classifier, model_type= model_type, image_size= image_size, with_outlier= with_outlier)
    abn.load(name_folder_abn = name_folder, epoch= epoch)
    abn.test_risk(ood_dataset=ood_dataset)
    
if __name__ == "__main__":
    pass