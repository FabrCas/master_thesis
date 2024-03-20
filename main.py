import time
import subprocess
import os
import argparse
import torch as T


from experiments import CIFAR_VITEA_benchmark, CIFAR_VITEA_Abnormality_module
from bin_ViTClassifier import DFD_BinViTClassifier_v7
from bin_classifier import DFD_BinClassifier_v4
from ood_detection import Abnormality_module_ViT, Abnormality_module

def parse_arguments(modes=["exe","test","train"]):
    parser = argparse.ArgumentParser(description="Deepfake detection module")
    
    # system execution parameters definition
    parser.add_argument('--useGPU',     type=bool,  default=True,   help="Usage to make compitations on tensors")
    parser.add_argument('--verbose',    type=bool,  default=True,   help="Verbose execution of the application")
    parser.add_argument('--mode',       type=str,   default=modes[2],  help="Choose the mode between 'exe', 'test', 'train'")
    
    return check_arguments(parser.parse_args())

def check_arguments(args): 
    if(args.useGPU):
        try:
            assert T.cuda.is_available()
            if(args.verbose):
                print("cuda available -> " + str(T.cuda.is_available()))
                current_device = T.cuda.current_device()
                print("current device -> " + str(current_device))
                print("number of devices -> " + str(T.cuda.device_count()))
                
                try:
                    print("name device {} -> ".format(str(current_device)) + " " + str(T.cuda.get_device_name(0)))
                except Exception as e:
                    print("exception device [] -> ".format(str(current_device)) + " " + +str(e))
                
        except:
            raise NameError("Cuda is not available, please check and retry")
        
    if(args.mode):
        if not args.mode in ['exe', 'test', 'train']:
            raise ValueError("Not valid mode is selected, choose between 'exe' or 'test'")
            
    if(args.verbose):
        print("Current main arguments:")
        [print(str(k) +" -> "+ str(v)) for k,v in vars(args).items()]
    return args


def create_folders(args):
    folders = os.listdir("./")

    if not('data' in folders) and args.mode in ["test","train"]:
        os.makedirs("./data")
        print("please include the datasets in the data folder")

    if not("models" in folders):
        os.makedirs("./models")
        print("required training mode launch to build the model")

    if not("results" in folders) and args.mode in ["test","train"]:
        os.makedirs("./results")

    if not(os.path.exists("./setup")):
        os.makedirs("./setup")
	
def redirect_requirements(name_file = "requirements.txt"):
    # create file
    subprocess.run(["touch", os.path.join("./setup", name_file)])
    # redirect list of dependencies output
    subprocess.run(["pip freeze > {}".format(os.path.join("./setup", name_file))], shell= True)

    return 
		

def main():
    # parse main arguments
    args = parse_arguments()
    if args is None: exit()

    # create project folders
    create_folders(args)

    type_test = "abn_mix"
    
    if type_test == "test_benchmark":
        prog_model          = 3
        name_folder         = "train_3_DeiT_tiny_27-02-2024"
        epoch_classifier    = 50
        epoch_autoencodder  = 50
        classifier = CIFAR_VITEA_benchmark(cifar100=False, prog_model= prog_model)
        classifier.load_classifier(epoch=epoch_classifier, name_folder=name_folder)
        classifier.load_autoencoder(epoch=epoch_autoencodder, name_folder= name_folder)
        
        name_folder = "Abnormality_module_ViT_encoder_v3_test_outlier_01-03-2024"
        epoch = 20
        model_type= "encoder_v3"
        abn = CIFAR_VITEA_Abnormality_module(classifier, model_type= model_type, with_outlier= True)
        abn.load(name_folder_abn = name_folder, epoch= epoch)
        
        for a in ["mnist", "fmnist", "svhn", "dtd", "tiny_imagenet", "cifar100"]:
            abn.test_risk(ood_dataset=a)
            
    elif type_test == "df_content":
        bin_classifier = DFD_BinViTClassifier_v7(scenario = "content", useGPU= True, model_type= "ViTEA_timm", prog_pretrained_model= 3, train_together= False)
        bin_classifier.load("faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_13-02-2024", 25)
        bin_classifier.test()
        
    elif type_test == "df_group":
        bin_classifier = DFD_BinClassifier_v4(scenario = "group", useGPU= True, model_type= "Unet5_Scorer")
        bin_classifier.load("gan_Unet5_Scorer_v4_07-01-2024", 71)
        bin_classifier.test()
        
    elif type_test == "df_mix":
        bin_classifier = DFD_BinClassifier_v4(scenario = "mix", useGPU= True, model_type= "Unet5_Residual_Scorer")
        bin_classifier.load("m_0_Unet5_Residual_Scorer_v4_02-03-2024", 51)
        bin_classifier.test()
        
    elif type_test == "abn_content_synth":
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
        
        abn = Abnormality_module_ViT(classifier, scenario=scenario, model_type= "encoder_v4")
        abn.load("Abnormality_module_ViT_encoder_v4_224p_50epochs_15-02-2024", 50)
        abn.test_risk()
        
    elif type_test == "abn_content":
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
        
        abn = Abnormality_module_ViT(classifier, scenario=scenario, model_type= "encoder_v3")
        abn.load("Abnormality_module_ViT_encoder_v3_224p_fullExtendedOOD_14-02-2024", 20)
        abn.test_risk()
        
    elif type_test == "abn_group": 
        scenario = "group"
        classifier_name = "gan_Unet5_Scorer_v4_07-01-2024"
        classifier_type = "Unet5_Scorer"
        classifier_epoch = 71
        classifier = DFD_BinClassifier_v4(scenario=scenario, model_type=classifier_type)
        classifier.load(classifier_name, classifier_epoch)
        resolution = "224p"
        
        abn = Abnormality_module(classifier, scenario=scenario, model_type= "encoder_v3")
        abn.load("Abnormality_module_encoder_v3_224p_50e_fullExtendedOOD_05-03-2024", 50)
        abn.test_risk()

    elif type_test == "abn_mix": 
        scenario = "mix"
        classifier_name = "m_0_Unet5_Residual_Scorer_v4_02-03-2024"
        classifier_type = "Unet5_Residual_Scorer"
        classifier_epoch = 51
        classifier = DFD_BinClassifier_v4(scenario=scenario, model_type=classifier_type)
        classifier.load(classifier_name, classifier_epoch)
        resolution = "224p"
    
        abn = Abnormality_module(classifier, scenario=scenario, model_type= "encoder_v3")
        abn.load("Abnormality_module_encoder_v3_224p_50e_fullExtendedOOD_06-03-2024", 50)
        abn.test_risk()
        
            

	
if __name__ == "__main__":
    print("started execution {}".format(os.getcwd()+ "/main.py"))
    t_start = time.time()
    main()
    redirect_requirements()
    exe_time = time.time() - t_start
    print("Execution time: {} [s]".format(round(exe_time,5)))

