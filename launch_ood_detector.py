from ood_detection import * 
    
if __name__ == "__main__":
    #                           [Start test section] 
    
    # [1] load deep fake classifier: choose classifier model as module A associated with a certain scenario
    classifier_model = 4
    scenario = None

    if classifier_model == 0:      # Unet + classifiier Faces
        
        scenario = "content"
        classifier_name = "faces_Unet4_Scorer112p_v4_03-12-2023"
        classifier_type = "Unet4_Scorer"
        classifier_epoch = 73
        classifier = DFD_BinClassifier_v4(scenario=scenario, model_type=classifier_type)
        classifier.load(classifier_name, classifier_epoch)
        resolution = "112p"
    
    elif classifier_model == 1:   # Unet + classifiier + confidence
        
        scenario = "content"
        classifier_name = "faces_Unet4_Scorer_Confidence_112p_v5_02-01-2024"
        classifier_type = "Unet4_Scorer_Confidence"
        classifier_epoch = 98
        classifier = DFD_BinClassifier_v5(scenario=scenario, model_type=classifier_type)
        classifier.load(classifier_name, classifier_epoch)
        conf_usage_mode = "ignore"  # choose btw: ignore, merge or alone
        resolution = "112p"
    
    elif classifier_model == 2:         # ViT + Autoencoder
        
        scenario = "content"
        classifier_name     = "faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_13-02-2024"
        classifier_type     = "ViTEA_timm"
        autoencoder_type    = "vae"
        prog_model_timm     = 3 # (tiny DeiT)
        classifier_epoch    = 25
        autoencoder_epoch   = 25
        classifier = DFD_BinViTClassifier_v7(scenario=scenario, model_type=classifier_type, model_ae_type = autoencoder_type,\
                                             prog_pretrained_model= prog_model_timm)
        # load classifier & autoencoder
        classifier.load_both(classifier_name, classifier_epoch, autoencoder_epoch)
        resolution = "224p"
    
    elif classifier_model == 3:   # Unet + classifiier GAN
        scenario = "group"
        classifier_name = "gan_Unet5_Scorer_v4_07-01-2024"
        classifier_type = "Unet5_Scorer"
        classifier_epoch = 71
        classifier = DFD_BinClassifier_v4(scenario=scenario, model_type=classifier_type)
        classifier.load(classifier_name, classifier_epoch)
        resolution = "224p"
        
    elif classifier_model == 4:   # Unet + classifiier Mix
        scenario = "mix"
        classifier_name = "m_0_Unet5_Residual_Scorer_v4_02-03-2024"
        classifier_type = "Unet5_Residual_Scorer"
        classifier_epoch = 51
        classifier = DFD_BinClassifier_v4(scenario=scenario, model_type=classifier_type)
        classifier.load(classifier_name, classifier_epoch)
        resolution = "224p"
    
    
    add2name = lambda name, add_name: name + "_" + add_name if add_name!="" else name
    
# ________________________________ baseline  _______________________________________
def test_baseline_implementation():
    ood_detector = Baseline(classifier= None, id_data_test = None, ood_data_test = None, useGPU= True)
    ood_detector.verify_implementation()
    
def test_baseline_facesCDDB_CIFAR():
    # select executions
    exe = [1,0]
            
    # [2] define the id/ood data
    
    # id_data_train    = CDDB_binary(train = True, augment = False
    # laod id data test
    id_data_train      = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment = False)
    id_data_test       = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment = False)
    _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
    # ood_data_train   = getCIFAR100_dataset(train = True)
    ood_data_test    = getCIFAR100_dataset(train = False)
    
    
    # [3] define the detector
    ood_detector = Baseline(classifier=classifier, task_type_prog= 0, name_ood_data="cifar100", id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
    
    # [4] launch analyzer/training
    if exe[0]: ood_detector.test_probabilties()
    
    # [5] launch testing
    if exe[1]:
        # ood_detector.test_threshold(thr_type="fpr95_abnormality",   normality_setting=False)
        ood_detector.test_threshold(thr_type="fpr95_normality",     normality_setting=True)
    
def test_baseline_scenario(scenario = "content"):
    if scenario == "content":
        name_ood_data_content  = "CDDB_content_faces_scenario"
    elif scenario == "group":
        name_ood_data_content  = "CDDB_group_gan_scenario"
    elif scenario == "mix":
        name_ood_data_content  = "CDDB_mix_m_0_scenario"
        
    # laod id data test
    id_data_train      = CDDB_binary_Partial(scenario = scenario, train = True,  ood = False, augment = False)
    id_data_test       = CDDB_binary_Partial(scenario = scenario, train = False, ood = False, augment = False)
    _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
    
    # load ood data test
    ood_data_test  = CDDB_binary_Partial(scenario = scenario, train = False,  ood = True, augment = False)
    
    # print(len(id_data_test), len(ood_data_test))
            
    ood_detector = Baseline(classifier=classifier, task_type_prog = 0, name_ood_data = name_ood_data_content ,  id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
    
    ood_detector.test_probabilties()

    
# ________________________________ baseline + ODIN  ________________________________

def test_baselineOdin_facesCDDB_CIFAR():

    # [2] define the id/ood data
    
    # id_data_train    = CDDB_binary(train = True, augment = False
    # laod id data test
    id_data_train      = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment = False)
    id_data_test       = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment = False)
    _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
    # ood_data_train   = getCIFAR100_dataset(train = True)
    ood_data_test    = getCIFAR100_dataset(train = False)
    
    # [3] define the detector
    ood_detector = Baseline_ODIN(classifier=classifier, task_type_prog= 0, name_ood_data="cifar100", id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
    
    # [4] launch analyzer/training
    ood_detector.test_probabilties()

def test_baselineOdin_scenario(scenario = "content"):
    if scenario == "content":
        name_ood_data_content  = "CDDB_content_faces_scenario"
    elif scenario == "group":
        name_ood_data_content  = "CDDB_group_gan_scenario"
    elif scenario == "mix":
        name_ood_data_content  = "CDDB_mix_m_0_scenario"

    # laod id data test
    id_data_train      = CDDB_binary_Partial(scenario = scenario, train = True,  ood = False, augment = False)
    id_data_test       = CDDB_binary_Partial(scenario = scenario, train = False, ood = False, augment = False)
    _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
    
    # load ood data test
    ood_data_test  = CDDB_binary_Partial(scenario = scenario, train = False,  ood = True, augment = False)
    
    ood_detector = Baseline_ODIN(classifier=classifier, task_type_prog = 0, name_ood_data = name_ood_data_content ,  id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
    
    ood_detector.test_probabilties()

# ______________________________ confidence detector  _______________________________
def test_confidenceDetector_facesCDDB_CIFAR():

    # [2] define the id/ood data
    
    # id_data_train    = CDDB_binary(train = True, augment = False
    # laod id data test
    id_data_train      = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment = False)
    id_data_test       = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment = False)
    _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
    # ood_data_train   = getCIFAR100_dataset(train = True)
    ood_data_test    = getCIFAR100_dataset(train = False)
    
    # [3] define the detector
    ood_detector = Confidence_Detector(classifier=classifier, task_type_prog= 0, name_ood_data="cifar100", id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
    
    # [4] launch analyzer/training
    ood_detector.test_confidence()

def test_confidenceDetector_content_faces():
    name_ood_data_content  = "CDDB_content_faces_scenario"

    # laod id data test
    id_data_train      = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment = False)
    id_data_test       = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment = False)
    _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
    
    # load ood data test
    ood_data_test  = CDDB_binary_Partial(scenario = "content", train = False,  ood = True, augment = False)
    
    ood_detector = Confidence_Detector(classifier=classifier, task_type_prog = 0, name_ood_data = name_ood_data_content ,  id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
    
    ood_detector.test_confidence()

# ________________________________ abnormality module  _____________________________

def train_abn_basic():        
    abn = Abnormality_module(classifier, scenario = "content", model_type="basic")
    abn.train(additional_name="112p", test_loop=True)
    
    # x = T.rand((1,3,112,112)).to(abn.device)
    # y = abn.forward(x)
    # print(y)

def train_abn_encoder(type_encoder = "encoder_v3", add_name = "", att_map_mode = "residual"):
    """
    Args:
        att_map_mode (str, optional): Choose the attention map used as auxiliary data:
        "residual", use residual as: (cls_attention_map - rec_cls_attention_map)^2.
        "cls_attention_map", use only the cls_attention_map as latent attention representation
        "cls_rec_attention_maps" use both cls_attention_map and its reconstruction (stacked)
        "full_cls_attention_maps", use cls_attention_map, and the attention map over patches (stacked)
        "full_cls_rec_attention_maps", use cls_attention_map, its reconstruction and the attention map over patches (stacked)
        "residual_full_attention_map", use residual and attention map of pathes (stacked)
        . Defaults to "residual".
    """
    
    
    if classifier_model == 2:
        abn = Abnormality_module_ViT(classifier, scenario = scenario, model_type= type_encoder, att_map_mode=att_map_mode)
    else: 
        abn = Abnormality_module(classifier, scenario = scenario, model_type= type_encoder, conf_usage_mode = conf_usage_mode)
    # abn.train(additional_name= resolution + "_ignored_confidence" , test_loop=False)
    abn.train(additional_name= add2name(resolution, add_name)  , test_loop=False)
    
def train_extended_abn_encoder(type_encoder = "encoder_v3", add_name = ""):
    """ uses extended OOD data"""
    if classifier_model == 2:
        abn = Abnormality_module_ViT(classifier, scenario = scenario, model_type= type_encoder, extended_ood = True)
    else: 
        abn = Abnormality_module(classifier, scenario = scenario, model_type= type_encoder, extended_ood = True,  conf_usage_mode = conf_usage_mode)
    abn.train(additional_name= add2name(resolution, add_name) + "_extendedOOD", test_loop=False)
    
def train_nosynt_abn_encoder(type_encoder = "encoder_v3", add_name = ""):
    """ uses extended OOD data"""
    
    if classifier_model == 2:
        abn = Abnormality_module_ViT(classifier, scenario = scenario, model_type= type_encoder, use_synthetic= False)
    else: 
        abn = Abnormality_module(classifier, scenario = scenario, model_type=type_encoder, use_synthetic= False,  conf_usage_mode = conf_usage_mode)
    abn.train(additional_name= add2name(resolution, add_name) + "_nosynt", test_loop=False)

def train_full_extended_abn_encoder(type_encoder = "encoder_v3", add_name = ""):
    
    """ uses extended OOD data"""
    if classifier_model == 2:
        abn = Abnormality_module_ViT(classifier, scenario = scenario, model_type= type_encoder, extended_ood = True, balancing_mode="all")
    else: 
        abn = Abnormality_module(classifier, scenario = scenario, model_type= type_encoder, extended_ood = True, balancing_mode="all",  conf_usage_mode = conf_usage_mode)
    abn.train(additional_name = add2name(resolution, add_name) + "_fullExtendedOOD", test_loop=False)

def test_abn(name_model, epoch, type_encoder = "encoder_v3", att_map_mode = "residual"):
    
    def test_forward():
        dl =  DataLoader(abn.dataset_train, batch_size = 16,  num_workers = 8,  shuffle= True,   pin_memory= False)
        for x,y in dl:
            x = x.to(abn.device)
            # output abnormality module
            out_abn = abn.forward(x)
            
            
            # output softmax baseline
            out_soft, _, _ = abn._forward_A(x)
            out_soft, _ = T.max(out_soft, dim = 1)
            
            for i in range(16):
                
                y_abn = round(T.squeeze(out_abn[i]).item(),4)
                y_soft = round(out_soft[i].item(),4)
                                    
                label = y[i].tolist() == [1,0]
                
                showImage(x[i], name = "ID: "+ str(label) + " abn: " + str(y_abn) + " soft: " + str(y_soft) + " ")
                print(y_abn, y_soft)
                
                
            # print(y)
            break
    
    
    # load model
    if classifier_model == 2:
        abn = Abnormality_module_ViT(classifier, scenario=scenario, model_type= type_encoder, att_map_mode = att_map_mode)
    else: 
        abn = Abnormality_module(classifier, scenario=scenario, model_type=type_encoder,  conf_usage_mode = conf_usage_mode)
        
    abn.load(name_model, epoch)
    
    # test_forward()

    # launch test with non-thr metrics
    abn.test_risk()
