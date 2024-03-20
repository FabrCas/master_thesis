from bin_classifier import * 
    
    
def test_workers_dl():                
    dataset = CDDB_binary()
    test_num_workers(dataset, batch_size  =32)   # use n_workers = 8

def test_cutmix1():
    
    cutmix = v2.CutMix(num_classes=2)

    def collate_cutmix(batch):
        """
        this function apply the CutMix technique with a certain probability (half probability)
        the batch should be defined with idx labels, but cutmix returns a sequence of values (n classes) for each label
        based on the composition
        """
        prob = 0.5
        if random .random() < prob:
            return cutmix(*default_collate(batch))
        else:
            return default_collate(batch)
    
    dl = DataLoader(CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment= False, label_vector= False), batch_size=8, shuffle= True, collate_fn= collate_cutmix)
    # classifier = BinaryClassifier(useGPU=True, batch_size=8, model_type = "")
    
    show = False
    
    for idx_out, (x,y) in enumerate(dl):
        # x,y torch tensor 
        print(x.shape, y.shape)
        
        # since we are using cutmix and "label_vector"= False in the Dataset parameters, adjust y if is the case
        if len(y.shape) == 1:
            y = T.nn.functional.one_hot(y)
        
        # x_ ,y_ = cutmix(x,y)
        
        for idx in range(len(x)):
            print(y[idx])
            if show :
                showImage(x[idx])
            
        if idx_out > 5: break

def test_cutmix2():
    dl = DataLoader(CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment= False, label_vector=True), batch_size=2, shuffle= True)
    classifier = BinaryClassifier(useGPU=True, batch_size=8, model_type = "")
    
    show = True
    
    for idx_out, (x,y) in enumerate(dl):
        # x,y torch tensor 
        # print(x.shape, y.shape)
    
        x_ ,y_ = classifier.cutmix_custom(x.numpy(),y.numpy(), prob= 0.5, verbose= True)
        
        for idx in range(len(x)):
            print("original labels: ",y[idx]," new labels: ", y_[idx])
            if show :
                showImage(x_[idx])
        print()
            
        if idx_out > 5: break   
        
# ________________________________ v1  ________________________________

def test_binary_classifier_v1():
    bin_classifier = DFD_BinClassifier_v1(useGPU = True, model_type="resnet_pretrained")
    print(bin_classifier.device)
    bin_classifier.getLayers(show = True)
    
def train_v1():
    bin_classifier = DFD_BinClassifier_v1(useGPU = True, model_type="resnet_pretrained")
    bin_classifier.load("resnet50_ImageNet_13-10-2023", 20)
    bin_classifier.train(name_train="resnet50_ImageNet")
    
def test_v1_metrics():
    bin_classifier = DFD_BinClassifier_v1(useGPU = True, model_type="resnet_pretrained")
    bin_classifier.load("resnet50_ImageNet_13-10-2023", 20)
    bin_classifier.test()

# ________________________________ v2  ________________________________

def test_binary_classifier_v2():
    bin_classifier = DFD_BinClassifier_v2(scenario = "content", useGPU= True, model_type="resnet_pretrained")
    # bin_classifier.load("resnet50_ImageNet_13-10-2023", 20)
    # bin_classifier.getLayers(show = True)
    
    train_dataset  = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment= True)
    test_dataset   = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment= False)
    valid_dataset, test_dataset = sampleValidSet(trainset= train_dataset, testset= test_dataset, useOnlyTest = True, verbose = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size= 32, num_workers= 8, shuffle= False, pin_memory= True)
    print(bin_classifier.valid(epoch=0, valid_dataloader= valid_dataloader))
    
def train_v2_content_scenario():
    bin_classifier = DFD_BinClassifier_v2(scenario = "content", useGPU= True, model_type="resnet_pretrained")
    bin_classifier.early_stopping_trigger = "acc"
    bin_classifier.train(name_train= "faces_resnet_50ImageNet")   #name with the pattern {data scenario}_{model name}, the algorithm include other name decorations
        
def train_v2_group_scenario():
    bin_classifier = DFD_BinClassifier_v2(scenario = "group", useGPU= True, model_type="resnet_pretrained")
    bin_classifier.early_stopping_trigger = "acc"
    bin_classifier.train(name_train="group_resnet50_ImageNet")   #name with the pattern {data scenario}_{model name}, the algorithm include other name decorations

def train_v2_mix_scenario():
    bin_classifier = DFD_BinClassifier_v2(scenario = "mix", useGPU= True, model_type="resnet_pretrained")
    bin_classifier.early_stopping_trigger = "acc"
    bin_classifier.train(name_train="mix_resnet50_ImageNet")   #name with the pattern {data scenario}_{model name}, the algorithm include other name decorations

def test_v2_metrics(name_model, epoch, scenario):
    bin_classifier = DFD_BinClassifier_v2(scenario = scenario, useGPU= True, model_type="resnet_pretrained")
    bin_classifier.load(name_model, epoch)
    bin_classifier.test()

# ________________________________ v3  ________________________________

def test_binary_classifier_v3():
    bin_classifier = DFD_BinClassifier_v3(scenario = "content", useGPU= True)
    # bin_classifier.train(name_train= "faces_resnet50EDS")

def train_v3_scenario():
    bin_classifier = DFD_BinClassifier_v3(scenario = data_scenario, useGPU= True)
    bin_classifier.train(name_train= scenario_setting + "_resnet50EDS")

def test_v3_metrics(name_model, epoch):
    bin_classifier = DFD_BinClassifier_v3(scenario = data_scenario, useGPU= True)
    bin_classifier.load(name_model, epoch)
    bin_classifier.test()  

def showReconstruction_v3(name_model, epoch):
    bin_classifier = DFD_BinClassifier_v3(scenario = data_scenario, useGPU= True)
    bin_classifier.load(name_model, epoch)
    img, _ = bin_classifier.test_dataset.__getitem__(300)
    print(img.shape)
    showImage(img)
    img     = T.unsqueeze(img, dim= 0).to(bin_classifier.device)
    enc     = bin_classifier.model.encoder_module.forward(img)
    rec_img = bin_classifier.model.decoder_module.forward(enc)
    logits  = bin_classifier.model.scorer_module.forward(enc)
    rec_img = T.squeeze(rec_img, dim = 0)
    print(rec_img)
    print(logits)
    showImage(rec_img)

# ________________________________ v4  ________________________________

def train_v4_scenario(model_type, add_name =""):
    bin_classifier = DFD_BinClassifier_v4(scenario = data_scenario, useGPU= True, model_type=model_type)
    if add_name != "":
        bin_classifier.train(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = False)
    else:
        bin_classifier.train(name_train= scenario_setting + "_" + model_type, test_loop = False)

def test_v4_metrics(name_model, epoch, model_type):
    bin_classifier = DFD_BinClassifier_v4(scenario = data_scenario, useGPU= True, model_type= model_type)
    bin_classifier.load(name_model, epoch)
    bin_classifier.test()

def train_test_v4_scenario(model_type, add_name =""):
    bin_classifier = DFD_BinClassifier_v4(scenario = data_scenario, useGPU= True, model_type=model_type)
    if add_name != "":
        bin_classifier.train(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = False)
    else:
        bin_classifier.train(name_train= scenario_setting + "_" + model_type, test_loop = False)

def showReconstruction_v4(name_model, epoch, model_type, save = False):
    
    random.seed(datetime.now().timestamp())
    
    bin_classifier = DFD_BinClassifier_v4(scenario = data_scenario, useGPU= True,  model_type = model_type)
    bin_classifier.load(name_model, epoch)
    idx = random.randint(0, len(bin_classifier.test_dataset))
    img, _ = bin_classifier.test_dataset.__getitem__(idx)
    showImage(img, save_image= save)
    
    img     = T.unsqueeze(img, dim= 0).to(bin_classifier.device)
    logits, rec_img, enc = bin_classifier.model.forward(img)
    rec_img = T.squeeze(rec_img, dim = 0)
    print(logits)
    showImage(rec_img, save_image=save) 
    
# ________________________________ v5  ________________________________ confidence

def train_v5_scenario(model_type, add_name =""):
    bin_classifier = DFD_BinClassifier_v5(scenario = data_scenario, useGPU= True, model_type=model_type)
    if add_name != "":
        bin_classifier.train(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = False)
    else:
        bin_classifier.train(name_train= scenario_setting + "_" + model_type, test_loop = False)

def test_v5_metrics(name_model, epoch, model_type):
    bin_classifier = DFD_BinClassifier_v5(scenario = data_scenario, useGPU= True, model_type= model_type)
    bin_classifier.load(name_model, epoch)
    bin_classifier.test()


if __name__ == "__main__":
    #                           [Start test section] 
    
    scenario_prog       = 2
    data_scenario       = None
    
    if scenario_prog == 0: 
        data_scenario = "content"
    elif scenario_prog == 1:
        data_scenario = "group"
    elif scenario_prog == 2:
        data_scenario = "mix"
    
    scenario_setting = getScenarioSetting()[data_scenario]    # type of content/group