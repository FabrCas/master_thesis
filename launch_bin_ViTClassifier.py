from bin_ViTClassifier import * 
    
# ________________________________ v6  ________________________________

def train_v6_scenario(model_type, add_name ="", patch_size = None, emb_size = None):
    
    bin_classifier = DFD_BinViTClassifier_v6(scenario = data_scenario, patch_size= patch_size, emb_size= emb_size,
                                                useGPU= True, model_type=model_type)
    if add_name != "":
        bin_classifier.train(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = False)
    else:
        bin_classifier.train(name_train= scenario_setting + "_" + model_type, test_loop = False)

def test_v6_metrics(name_model, epoch, model_type, patch_size = None, emb_size = None):
    bin_classifier = DFD_BinViTClassifier_v6(scenario = data_scenario, patch_size= patch_size, emb_size= emb_size,
                                                useGPU= True, model_type= model_type)
    bin_classifier.load(name_model, epoch)
    bin_classifier.test()
    
def train_test_v6_metrics(model_type, add_name ="", patch_size = None, emb_size = None):
    bin_classifier = DFD_BinViTClassifier_v6(scenario = data_scenario, patch_size= patch_size, emb_size= emb_size,
                                                useGPU= True, model_type=model_type)
    if add_name != "":
        bin_classifier.train_and_test(name_train= scenario_setting + "_" + model_type + "_" + add_name)
    else:
        bin_classifier.train_and_test(name_train= scenario_setting + "_" + model_type)
        
# ________________________________ v7  ________________________________

# ------- attention map test

def test_generic_attention_map():

    from PIL import Image
    import matplotlib.pyplot as plt
    from timm.models import create_model
    from torchvision import transforms
    from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

    def to_tensor(img):
        transform_fn = Compose([Resize(249, 3), CenterCrop(224), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        return transform_fn(img)

    def show_img(img):
        img = np.asarray(img)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def show_img2(img1, img2, alpha=0.8):
        img1 = np.asarray(img1)
        img2 = np.asarray(img2)
        plt.figure(figsize=(10, 10))
        plt.imshow(img1)
        plt.imshow(img2, alpha=alpha)
        plt.axis('off')
        plt.show()

    def my_forward_wrapper(attn_obj):
        def my_forward(x):
            B, N, C = x.shape
            print("B: ", B)
            print("N: ", N)
            print("C: ", C)
            
            qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
            
            print("qkv shape", qkv.shape)
            
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
            
            print("q shape", q.shape)
            print("k shape", k.shape)
            print("v shape", v.shape)
            

            attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
            attn = attn.softmax(dim=-1)
            attn = attn_obj.attn_drop(attn)
            
            # print(attn.shape)
            
            attn_obj.attn_map = attn
            # attn_obj.cls_attn_map = attn[:, :, 0, 2:]
            attn_obj.cls_attn_map = attn[:, :, 0, 1:]

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = attn_obj.proj(x)
            x = attn_obj.proj_drop(x)
            return x
        return my_forward
    
    bin_classifier = DFD_BinViTClassifier_v7(scenario = "content", useGPU= True)
    data_iter = bin_classifier.train_dataset
    x1, y = data_iter.__getitem__(0)
    x2, _ = data_iter.__getitem__(1)
    

    
    # model = create_model('deit_small_distilled_patch16_224', pretrained=True)
    model = create_model(model_name='vit_base_patch16_224.augreg_in21k' , pretrained=True, num_classes=2, drop_rate=0.1)
    
    # bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type)
    # bin_classifier.load(name_model, epoch)
    
    
    model.blocks[-1].attn.forward = my_forward_wrapper(model.blocks[-1].attn)
    # model = bin_classifier.model.model_vit

    x = T.stack((x1,x2))
    print("image batch shape: ", x.shape)
    
    # x = x.unsqueeze(0)

    _ = model(x)
    
    att_map = model.blocks[-1].attn.attn_map.mean(dim=1).detach()
    att_map = att_map[:, 1:, 1:].view(-1,1,196,196)
    print("full attention map: ", att_map.shape)
    
    # print(model.blocks[-1].attn.cls_attn_map.mean(dim=1).shape)
    att_map_cls = model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(-1, 14, 14).detach()
    att_map_cls = att_map_cls.unsqueeze(dim = 1)
    
    print("attention map cls (batch) shape: ", att_map_cls.shape)
    
    # print(model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(-1,14,14).shape)
    # cls_weight = model.blocks[-1].attn.cls_attn_map[:].mean(dim=1).view(14, 14).detach()
    # print(cls_weight.shape)
    
    
    print("att_map max: ", T.max(att_map))
    print("att_map min: ", T.min(att_map))
    print("att_map cls max: ", T.max(att_map_cls))
    print("att_map cls min: ", T.min(att_map_cls))
    
    att_map_up          = F.interpolate(att_map, (224,224), mode = "bilinear")
    att_map_cls_up      = F.interpolate(att_map_cls, (224, 224), mode='bilinear')
    
    

    x_show          = (x[0].permute(1, 2, 0) + 1)/2
    
    att_map         = att_map[0].permute(1,2,0)
    att_map_up      = att_map_up[0].permute(1,2,0)
    att_map_cls     = att_map_cls[0].permute(1,2,0)
    att_map_cls_up  = att_map_cls_up[0].permute(1,2,0)
    
    
    show_img(x_show)
    
    
    show_img(att_map)
    show_img(att_map_up)
    
    show_img(att_map_cls)
    show_img(att_map_cls_up)
    
    print(x_show.shape)
    print(att_map_cls_up.shape)
    
    show_img2(x_show, att_map_cls_up, alpha=0.8)

def test_attention_map_v7(name_model, epoch, epoch_ae, prog_model = 3, model_type = "ViTEA_timm", autoencoder_type = "vae"):
    
    bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type,\
                                                transform_prog=0, prog_pretrained_model=prog_model, model_ae_type= autoencoder_type)
    
    bin_classifier.load_both(name_model, epoch, epoch_ae= epoch_ae)
    data_iter = bin_classifier.test_dataset
    save    = True
    img_id  = 0
    
    path_save_images = os.path.join(bin_classifier.path_models, bin_classifier.classifier_name)
    
    print(f"path save for images: {path_save_images}")
    
    img, y = data_iter.__getitem__(img_id)
    
    # img = trans_input_base()(Image.open("./static/test_image_attention.png"))
    
    showImage(img, save_image= save, name="attention_original_" + str(img_id), path_save=path_save_images)

    img = img.unsqueeze(dim=0)

    img = img.to(device = bin_classifier.device)
    
    out = bin_classifier.model.forward(img)
    
    att_map         = out[2]
    att_map_patches = out[3]
    
    print("att_map cls: shape, max, min: ", att_map.shape, T.max(att_map), T.min(att_map))
    
    showImage(att_map[0], has_color= False, save_image= save, name="attention_map_" + str(img_id), path_save=path_save_images)
    
    print("att_map patches: shape, max, min: ", att_map_patches.shape, T.max(att_map_patches), T.min(att_map_patches))
    
    showImage(att_map_patches[0], has_color= False, save_image= save, name="attention_map_patches_" + str(img_id), path_save=path_save_images)
    
    blended_results, _  = include_attention(img, att_map, alpha= 0.7)
    
    showImage(blended_results[0], save_image= save, name="attention_fused_" + str(img_id), path_save=path_save_images)
    
    show_imgs_blend(img[0].cpu(), att_map[0].cpu(), alpha=0.8, save_image= save, name="attention_blend_" + str(img_id), path_save=path_save_images)
    
    # reconstruction 
    # if bin_classifier.model_ae_type == "vae":
    #     rec_att_map, _, _  = bin_classifier.autoencoder.forward(att_map)
    # else:    
    rec_att_map = bin_classifier.autoencoder.forward(att_map)
            
    print("rec_att_map: shape, max, min: ", rec_att_map.shape, T.max(rec_att_map), T.min(rec_att_map))

    showImage(rec_att_map[0], has_color= False, save_image= save, name="attention_map_AE_" + str(img_id), path_save=path_save_images)
    

# ------- train

def train_v7_scenario(prog_vit, model_type = "ViTEA_timm", add_name ="", autoencoder_type = "vae"):
    
    bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type=model_type,\
                                            prog_pretrained_model= prog_vit, model_ae_type= autoencoder_type)
    if add_name != "":
        bin_classifier.train(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = False)
    else:
        bin_classifier.train(name_train= scenario_setting + "_" + model_type, test_loop = False)

def train_v7_scenario_separately(prog_vit, model_type = "ViTEA_timm", add_name ="", autoencoder_type = "vae", test_loop = False):
    
    bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type=model_type,\
                                                train_together=False, prog_pretrained_model= prog_vit, model_ae_type= autoencoder_type)
    
    if add_name != "":
        bin_classifier.train(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = test_loop)
    else:
        bin_classifier.train(name_train= scenario_setting + "_" + model_type, test_loop = test_loop)

def trainViT_v7_scenario(prog_vit, model_type = "ViTEA_timm", add_name ="", test_loop = False):
    bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type=model_type,\
                                        train_together=False, prog_pretrained_model= prog_vit)
    
    if add_name != "":
        bin_classifier.trainViT(name_train= scenario_setting + "_" + model_type + "_" + add_name, test_loop = test_loop)
    else:
        bin_classifier.trainViT(name_train= scenario_setting + "_" + model_type, test_loop = test_loop)

def continueTrainViT_v7_scenario(name_folder, prog_vit, start_epoch = 0, end_epoch = None, model_type = "ViTEA_timm", test_loop = False):
    bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type=model_type,\
                                        train_together=False, prog_pretrained_model= prog_vit)
    
    bin_classifier.continueTrainViT(name_folder=name_folder, start_epoch= start_epoch, end_epoch=end_epoch, test_loop = test_loop)

    
def trainAE_v7_scenario(name_folder_train, epoch_vit, prog_vit, model_type = "ViTEA_timm", autoencoder_type = "vae", test_loop = False):
        bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type=model_type,\
                                    train_together=False, prog_pretrained_model= prog_vit, model_ae_type= autoencoder_type)
        
        bin_classifier.load(name_folder_train, epoch_vit)
        
        bin_classifier.trainAE(name_folder=name_folder_train, test_loop=test_loop)
    


# ------- test
    
def test_v7_metrics(name_folder_train, epoch, prog_model = 3, model_type = "ViTEA_timm"):
    bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type, prog_pretrained_model= prog_model, train_together= False)
    bin_classifier.load(name_folder_train, epoch)
    bin_classifier.test()

def test_v7_ae_metrics(name_folder_train, epoch, epoch_ae, prog_model = 3, model_type = "ViTEA_timm", autoencoder_type = "vae"):
    bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type, model_ae_type=autoencoder_type, 
                                                prog_pretrained_model= prog_model, train_together= False)
    
    bin_classifier.load_both(name_folder_train,epoch=epoch, epoch_ae= epoch_ae)
    
    bin_classifier.test_ae()

def test_v7_both_metrics(name_folder_train, epoch, epoch_ae, prog_model  = 3, model_type = "ViTEA_timm", autoencoder_type = "vae"):
    bin_classifier = DFD_BinViTClassifier_v7(scenario = data_scenario, useGPU= True, model_type= model_type, model_ae_type=autoencoder_type, 
                                        prog_pretrained_model= prog_model, train_together= False)
    
    bin_classifier.load_both(name_folder_train,epoch=epoch, epoch_ae= epoch_ae)
    
    bin_classifier.test()
    bin_classifier.test_ae()

   
   
if __name__ == "__main__":
    #                           [Start test section] 
    scenario_prog       = 1
    data_scenario       = None
    
    if scenario_prog == 0: 
        data_scenario = "content"
    elif scenario_prog == 1:
        data_scenario = "group"
    elif scenario_prog == 2:
        data_scenario = "mix"
    
    scenario_setting = getScenarioSetting()[data_scenario]    # type of content/group