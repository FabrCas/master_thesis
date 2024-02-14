import  os
import  torch               as T
import  numpy               as np
import  math
from    torch.nn            import functional as F
from    torch.utils.data    import DataLoader
from    torch.cuda.amp      import autocast
from    tqdm                import tqdm
from    datetime            import date
from    time                import time
from    sklearn.metrics     import precision_recall_curve, auc, roc_auc_score
from    torch.optim         import Adam, lr_scheduler
from    torch.cuda.amp      import GradScaler, autocast
# local import
from    dataset             import getScenarioSetting, CDDB_binary_Partial, CDDB_Partial, OOD_dataset, getCIFAR100_dataset, getMNIST_dataset, getFMNIST_dataset
from    experiments         import MNISTClassifier_keras
from    bin_classifier      import DFD_BinClassifier_v4, DFD_BinClassifier_v5
from    bin_ViTClassifier   import DFD_BinViTClassifier_v7
from    models              import Abnormality_module_Basic, Abnormality_module_Encoder_v1, Abnormality_module_Encoder_v2,\
                            Abnormality_module_Encoder_v3, Abnormality_module_Encoder_v4, Abnormality_module_Encoder_VIT_v3
from    utilities           import saveJson, loadJson, metrics_binClass, metrics_OOD, print_dict, showImage, check_folder, sampleValidSet, \
                            mergeDatasets, ExpLogger, loadModel, saveModel, duration, plot_loss, plot_valid

class OOD_Classifier(object):
    
    def __init__(self, id_data_test = None, ood_data_test = None, id_data_train = None, ood_data_train = None, useGPU = True):
        super(OOD_Classifier, self).__init__()
        
        
        # train data
        self.id_data_train  = id_data_train
        self.ood_data_train = ood_data_train
        # test sets
        self.id_data_test   = id_data_test
        self.ood_data_test  = ood_data_test
        
        # classifier types
        self.types_classifier = ["bin_class", "multi_class", "multi_label_class"]
        
        # general paths
        self.path_models    = "./models/ood_detection"
        self.path_results   = "./results/ood_detection"
        
        # execution specs
        self.useGPU = useGPU
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        
        # hyper-params
        self.batch_size     =  256  # 32 -> basic, 64/128 -> encoder
        
    
    #                                      data aux functions
    def compute_class_weights(self, verbose = False, positive = "ood", multiplier = 1, normalize = True, only_positive_weight = False):
        
        # TODO look https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        # to implement just pos weight computation 
        
        
        """ positive (str), if ood is used 1 to represent ood labels and 0 for id. The opposite behavior is obtained using "id" """
        
        print("\n\t\t[Computing class weights for the training set]\n")
        
        # set modality to load just the label
        self.dataset_train.set_only_labels(True)
        loader = DataLoader(self.dataset_train, batch_size= None,  num_workers = 8)  # self.dataset_train is instance of OOD_dataset class

        
        # compute occurrences of labels
        class_freq={}
        total = len(self.dataset_train)
        self.samples_train = total

    
        for y in tqdm(loader, total = len(loader)):
            
            # print(y.shape)
            # print(y.shape)
            # l = y.item()
            y = y.detach().cpu().tolist()
                
            if positive == "ood":
                l = y[1]
            elif positive == "id":
                l = y[0]
            else:
                ValueError('invalid value for the parameter "positive", choose between "ood  and "id"')
            
            if l not in class_freq.keys():
                class_freq[l] = 1
            else:
                class_freq[l] = class_freq[l]+1
        if verbose: print("class_freq -> ", class_freq)
        
        # compute the weights   
        class_weights = []
        for class_ in sorted(class_freq.keys()):
            freq = class_freq[class_]
            class_weights.append(round(total/freq,5))

        # normalize class weights with values between 0 and 1
        if normalize:
            max_value = max(class_weights)
            class_weights = [item/max_value for item in class_weights]
            
        # proportional increase over the weights
        if multiplier != 1:
            class_weights = [item * multiplier for item in class_weights]
        
        print("Class_weights-> ", class_weights)
                
        # turn back in loading modality sample + label
        self.dataset_train.set_only_labels(False)
        
        if only_positive_weight:
            print("Computing weight only for the positive class (label 1)")
            pos_weight = class_weights[1]/class_weights[0]
            print("positive class weight-> ", pos_weight)
            
            return [pos_weight]
        else:
            return class_weights 

    #                                       math/statistics aux functions
    
    # def weighted_binary_cross_entropy(output, target, weights=None):
        
    #     if weights is not None:
    #         assert len(weights) == 2
            
    #         loss = weights[1] * (target * T.log(output)) + \
    #             weights[0] * ((1 - target) * T.log(1 - output))
    #     else:
    #         loss = target * T.log(output) + (1 - target) * T.log(1 - output)

    #     return T.neg(T.mean(loss))
    
    def _sigmoid(self,x):
        """ numpy implementation of sigmoid actiavtion function"""
        return 1.0 / (1.0 + np.exp(-x))
        
    def _softmax(self,x):
        """ numpy implementation of softmax actiavtion function"""
        e = np.exp(x)
        return  e/e.sum(axis=1, keepdims=True)
        # return  e/e.sum()
        
    def entropy(self, distibution):
        """
            computes entropy from KL divergence formula, comparing the distribution with a uniform one    
        """
        return np.log(10.) + np.sum(distibution * np.log(np.abs(distibution) + 1e-11), axis=1, keepdims=True)
    
    def compute_statsProb(self, probability):
        """ get statistics from probability

        Args:
            probability (_type_):is the output from softmax-sigmoid function using the logits

        Returns:
            maximum_prob (np.array): max probability for each prediction (keepsdim = True)
            entropy (np.array): entropy from KL divergence with uniform distribution and data distribution
            mean_e (np.array): entropy's mean
            std_e (np.array): standard deviation entropy
        """
        # get the max probability
        maximum_prob    = np.max(probability, axis=1, keepdims= True)
        entropy         = self.entropy(probability)
        mean_e          = np.mean(entropy)
        std_e           = np.std(entropy)
        
        return maximum_prob, entropy, mean_e, std_e
    
    def compute_MSP(self, probabilities):     # should be not translated as direct measure of confidence
        """ computation of the baseline performance: maximum softmax performance (MSP)

        Args:
            probabilties (np.array): probabilities from logits

        Returns:
            confidences (np.array): list of confidences for each prediction
        """
        pred_value      = np.max(probabilities, -1)
        # add a fictitious dimension
        confidences     = np.reshape(pred_value, (len(pred_value), 1))
        
        return confidences
    
    def compute_avgMSP(self, probabilities):
        """_
            computes the average using max probabilities from test instances.
        """
        pred_value      = np.max(probabilities, -1)   # max among class probabilities
        return np.average(pred_value)
    
    # functions for ODIN framework    (https://arxiv.org/pdf/2109.14162v2.pdf)  # don't use during training of the model but just in the inference phase (forward + testing)

    def softmax_temperature(self, logits ,t = 1000):
        """ ODIN softmax scaled with temperature t to enhance separation between ID and OOD

        Args:
            x (numpy.ndarray or torch.Tensor): input array/tensor, sample or batch
            t (int, optional): Temperature scaling. Defaults to 1000.

        Returns:
            T: _description_
        """
        
        if isinstance(logits, T.Tensor):
            transform_back = True
            logits = logits.numpy()
        else:
            transform_back = False
        
        logits = logits/t
        prob = self._softmax(logits)
        if transform_back: 
            return T.tensor(prob)
        else:
            return prob
        
    def odin_perturbations(self,x, classifier, y = None, is_batch = True, loss_function = F.cross_entropy, epsilon = 12e-4, t = 1000):  # original hyperparms: epislon = 12e-4, t = 1000
        """ Perturbation from ODIN framework

        Args:
            x (T.tensor): input
            classifier (T.nn.Module): classifier module from pytorch framework
            is_batch (bool, optional): True if x is batch, False otherwise. Defaults to False.
            loss_function (function, optional): loss function to compute the loss for backpropagation. Defaults to F.cross_entropy.
            epsilon (float, optional): magnitude of perturbation. Defaults to 12e-4.

        Returns:
            perturbed_input (T.tensor): The perturbed input
        """
        
        # prepare
        try:
            classifier.model.eval()  # Set the model to evaluation mode
        except Exception as e:
            print(e)
        
        classifier.zero_grad()
        
        x = x.clone()
        x.requires_grad_(True)  # Enable gradient computation for the input
    
        
        with T.enable_grad():
            # forward and loss
            _, _, logits = classifier.forward(x)
            
            logits /= t
        
        if y is None:   # pseudo-labels
            # y = logits.max(dim = 1).indices  # same as 
            y  = T.argmax(logits, dim=-1)
        
        loss = loss_function(logits, y)
        loss.backward()
        
        # gradient = T.sign(x.grad.data)
        # Normalizing the gradient to binary in {0, 1}
        gradient = T.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        
        with T.no_grad():
            # perturbed_x = x - epsilon *gradient
            perturbed_x = T.add(x, -gradient, alpha= epsilon)
        
        perturbed_x.requires_grad = False
        classifier.zero_grad()
        return perturbed_x
    
    # metrics aux functions
    def compute_aupr(self, labels, pred):        
        p, r, _ = precision_recall_curve(labels, pred)
        return  auc(r, p)
    
    def compute_auroc(self, labels, pred):
        return roc_auc_score(labels, pred)
        
    def compute_curves(self, id_data, ood_data, positive_reversed = False):
        """_
            compute AUROC and AUPR, defining the vector of labels from the length of ID and OOD distribution
            can be chosen which label use as positive, defult is for ID data.
            set to False the parameter "positive_reversed" for abnormality detection (OOD).
        """
        
        # create the array with the labels initally full of zeros
        target = np.zeros((id_data.shape[0] + ood_data.shape[0]), dtype= np.int32)
        
       
        # print(target.shape)
        if positive_reversed:
            target[id_data.shape[0]:] += 1
        else:
            target[:id_data.shape[0]] += 1
       
        # print(target.shape)
        predictions = np.squeeze(np.vstack((id_data, ood_data)))
        # print(predictions.shape)
        
        aupr    = round(self.compute_aupr(target, predictions)*100, 2)
        auroc   = round(self.compute_auroc(target, predictions)*100, 2)
        print("\tAUROC(%)-> {}".format(auroc))
        print("\tAUPR (%)-> {}".format(aupr))
        
        return aupr, auroc

    def compute_metrics_ood(self, id_data, ood_data, path_save = None, positive_reversed = False):
        """_
            aux function used to compute fpr95, detection error and relative threshold.
            con be selected the positive label, the defulat one is for ID data, set to False
            with the parameter "positive_reversed" for abnormality detection (OOD).
        """
        target = np.zeros((id_data.shape[0] + ood_data.shape[0]), dtype= np.int32)
        
        if positive_reversed:
            target[id_data.shape[0]:] += 1
        else:
            target[:id_data.shape[0]] += 1
            
        # print(target)
        
        predictions = np.squeeze(np.vstack((id_data, ood_data)))
        # get metrics and save AUROC plot
        metrics_ood = metrics_OOD(targets=target, pred_probs= predictions, pos_label= 1, path_save_plot = path_save)
        
        return metrics_ood 
    
    # path utilities
    
    def get_path2SaveResults(self, train_name = None):
        """ Return the path to the folder to save results both for OOD metrics and ID/OOD bin classification"""
        
        # compose paths
        name_task                   = self.types_classifier[self.task_type_prog]
        path_results_task           = os.path.join(self.path_results, name_task)
        path_results_method         = os.path.join(path_results_task, self.name)
        path_results_ood_data       = os.path.join(path_results_method, "ood_" + self.name_ood_data)
        path_results_classifier     = os.path.join(path_results_ood_data, self.name_classifier) 
        if train_name is not None:   
            path_results_folder         = os.path.join(path_results_classifier,train_name)    
        
        # prepare file-system without forcing creations
        check_folder(path_results_task)
        check_folder(path_results_method)
        check_folder(path_results_ood_data)
        check_folder(path_results_classifier)
            
        
        if train_name is not None:
            # path_results_folder = check_folder(path_results_folder, force = True) #$
            check_folder(path_results_folder, force = False) #$
            return path_results_folder
        else:
            return path_results_classifier
    
    def get_path2SaveModels(self, train_name = None):
        
        # compose paths
        name_task                   = self.types_classifier[self.task_type_prog]
        path_models_task            = os.path.join(self.path_models, name_task)
        path_models_method          = os.path.join(path_models_task, self.name)
        path_models_ood_data        = os.path.join(path_models_method, "ood_"+self.name_ood_data)
        path_models_classifier      = os.path.join(path_models_ood_data, self.name_classifier)    
        if train_name is not None:   
            path_models_folder          = os.path.join(path_models_classifier, train_name)
        
        # prepare file-system without forcing creations
        check_folder(path_models_task)
        check_folder(path_models_method)
        check_folder(path_models_ood_data)
        check_folder(path_models_classifier)
        
        if train_name is not None:
            # path_models_folder = check_folder(path_models_folder, force = True)  #$
            check_folder(path_models_folder, force = False, is_model = True)  #$
            return path_models_folder
        else:
            return path_models_classifier
    
class Baseline(OOD_Classifier):             # No model training necessary (Empty model forlder)
    """
        OOD detection baseline using softmax probability
        
        IDEA: an higher MSP should be related to In-Distribution samples, a lower one instead to Out-Of-Distribution
        So values approching -> 1, are more likely to be ID, and the opposite with values approching 0.5
        
    """
    def __init__(self, classifier,  task_type_prog, name_ood_data, id_data_test , ood_data_test , id_data_train = None, ood_data_train = None, useGPU = True):
        """
        OOD_Baseline instantiation 

        Args:
            classifier (DFD_BinClassifier): classifier used for the main Deepfake detection task
            ood_data_test (torch.utils.data.Dataset): test set out of distribution
            ood_data_train (torch.utils.data.Dataset): train set out of distribution
            useGPU (bool, optional): flag to enable usage of GPU. Defaults to True.
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset used as ood data, is is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
                if None, the results will be not saved
        """
        super(Baseline, self).__init__(id_data_test = id_data_test, ood_data_test = ood_data_test,                  \
                                           id_data_train = id_data_train, ood_data_train = ood_data_train, useGPU = useGPU)
        # set the classifier
        self.classifier  = classifier
        self.name_classifier = self.classifier.classifier_name
        
        self.task_type_prog = task_type_prog
        self.name_ood_data  = name_ood_data
        
        # load the Pytorch dataset here
        try:
            self.dataset_test  = OOD_dataset(self.id_data_test, self.ood_data_test, balancing_mode = "max")
        except:
            print("Dataset data is not valid, please use instances of class torch.Dataset")
        
        # name of the classifier
        self.name = "baseline"
    
    #                                       testing functions
    def test_probabilties(self):
        """
            testing function using probabilty-based metrics, computing OOD metrics that are not threshold related
        """
        
        # saving folder path
        path_results_folder         = self.get_path2SaveResults()
        
        # define the dataloader 
        id_ood_dataloader   = DataLoader(self.dataset_test,  batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # use model on selected device
        self.classifier.model.to(self.device)
        
        # define empty list to store outcomes
        pred_logits = np.empty((0,2), dtype= np.float32)
        dl_labels = np.empty((0,2), dtype= np.int32)            # dataloader labels, binary one-hot encoding, ID -> [1,0], OOD -> [0,1]
        
        for idx, (x,y) in tqdm(enumerate(id_ood_dataloader), total= len(id_ood_dataloader)):
            
            # to test
            # if idx >= 10: break
            
            x = x.to(self.device)
            with T.no_grad():
                with autocast():
                    _ ,_, logits =self.classifier.forward(x)
                    
                    
            # to numpy array
            logits  = logits.cpu().numpy()
            y       = y.numpy()
                
            pred_logits = np.append(pred_logits, logits, axis= 0)
            dl_labels = np.append(dl_labels, y, axis= 0)
            
        # to softmax probabilities
        probs = self._softmax(pred_logits)

        # separation of id/ood labels and probabilities
        id_labels  =  dl_labels[:,0]                # filter by label column
        ood_labels =  dl_labels[:,1]
        prob_id     = probs[id_labels == 1]         # split forward probabilites between ID adn OOD, still a list of probabilities for each class learned by the model
        prob_ood    = probs[ood_labels == 1]
        

        # compute confidence (all)
        conf_all = round(self.compute_avgMSP(probs),3)
        print("Confidence ID+OOD\t{}".format(conf_all))
        
        maximum_prob_id,  entropy_id,  mean_e_id,  std_e_id  = self.compute_statsProb(prob_id)
        maximum_prob_ood, entropy_ood, mean_e_ood, std_e_ood = self.compute_statsProb(prob_ood)
        
        max_prob_id_mean    = np.mean(maximum_prob_id);     max_prob_id_std     = np.std(maximum_prob_id)
        max_prob_ood_mean   = np.mean(maximum_prob_ood);    max_prob_ood_std    = np.std(maximum_prob_ood)
        
        
        # in-out of distribution moments
        print("In-Distribution max prob         [mean (confidence ID),std]  -> ", max_prob_id_mean, max_prob_id_std)
        print("Out-Of-Distribution max prob     [mean (confidence OOD),std] -> ", max_prob_ood_mean, max_prob_ood_std)
        
        print("In-Distribution Entropy          [mean,std]                  -> ", mean_e_id, std_e_id)
        print("Out-Of-Distribution Entropy      [mean,std]                  -> ", mean_e_ood, std_e_ood)
        
        # normality detection
        print("Normality detection:")   # positive label -> ID data
        norm_base_rate = round(100*(prob_id.shape[0]/(prob_id.shape[0] + prob_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(norm_base_rate))
        print("\tKL divergence (entropy)")
        kl_norm_aupr, kl_norm_auroc = self.compute_curves(entropy_id, entropy_ood)
        print("\tPrediction probability")
        p_norm_aupr, p_norm_auroc = self.compute_curves(maximum_prob_id, maximum_prob_ood)

        
        # abnormality detection
        print("Abnormality detection:")   # positive label -> OOD data
        abnorm_base_rate = round(100*(prob_ood.shape[0]/(prob_id.shape[0] + prob_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(abnorm_base_rate))
        print("\tKL divergence (entropy)")
        kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-entropy_id, 1-entropy_ood, positive_reversed= True)
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-entropy_id, 1-entropy_ood, positive_reversed= True)
        print("\tPrediction probability")
        p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-maximum_prob_id, 1-maximum_prob_ood, positive_reversed= True)
        # p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-maximum_prob_id, 1-maximum_prob_ood, positive_reversed= True)

        # compute fpr95, detection_error and threshold_error 
        metrics_norm = self.compute_metrics_ood(maximum_prob_id, maximum_prob_ood, path_save = path_results_folder)
        print("OOD metrics:\n", metrics_norm)  
        metrics_abnorm = self.compute_metrics_ood(1-maximum_prob_id, 1-maximum_prob_ood, positive_reversed= True)
        print("OOD metrics:\n", metrics_abnorm)   
                     
        # store statistics/metrics in a dictionary
        data = {
            "ID_max_prob": {
                "mean":  float(max_prob_id_mean), 
                "var":   float(max_prob_id_std)
            },
            "OOD_max_prob": {
                "mean": float(max_prob_ood_mean), 
                "var":  float(max_prob_ood_std) 
            },
            "ID_entropy":   {
                "mean": float(mean_e_id), 
                "var":  float(std_e_id)
            },
            "OOD_entropy":  {
                "mean": float(mean_e_ood), 
                "var":  float(std_e_ood)
            },
            
            "normality": {
                "base_rate":        float(norm_base_rate),
                "KL_AUPR":          float(kl_norm_aupr),
                "KL_AUROC":         float(kl_norm_auroc),
                "Prob_AUPR":        float(p_norm_aupr),   
                "Prob_AUROC":       float(p_norm_auroc)
            },
            "abnormality":{
                "base_rate":        float(abnorm_base_rate),
                "KL_AUPR":          float(kl_abnorm_aupr),
                "KL_AUROC":         float(kl_abnorm_auroc),
                "Prob_AUPR":        float(p_abnorm_aupr),   
                "Prob_AUROC":       float(p_abnorm_auroc),

            },
            "avg_confidence":               float(conf_all),
            "fpr95_normality":              float(metrics_norm['fpr95']),
            "detection_error_normality":    float(metrics_norm['detection_error']),
            "threshold_normality":          float(metrics_norm['thr_de']),
            "fpr95_abnormality":            float(metrics_abnorm['fpr95']),
            "detection_error_abnormality":  float(metrics_abnorm['detection_error']),
            "threshold_abnormality":        float(metrics_abnorm['thr_de'])
            
            
        }
        
        # save data (JSON)
        if self.name_ood_data is not None:
            path_results_folder         = self.get_path2SaveResults()
            name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
            path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
            
            print(path_result_save)
                  
            saveJson(path_file = path_result_save, data = data)
    
    def verify_implementation(self):
        """
            Function used to verify the correct implementation of the baseline.
            Recreating the experiment carried out in the notebook: https://github.com/2sang/OOD-baseline/tree/master/notebooks
        """
    
        classifier = MNISTClassifier_keras()
        classifier.load_model()
               
        mnist_testset   = getMNIST_dataset(train = False)
        fmnist_testset  = getFMNIST_dataset(train = False)
        dataset_ood     = OOD_dataset(id_data= mnist_testset, ood_data=fmnist_testset, balancing_mode = None, grayscale= True)
        dataloader_ood  = DataLoader(dataset_ood,  batch_size= 128, num_workers= 8, shuffle= False, pin_memory= True)   # set shuffle to False, batch_size = 128
        
        # empty lists to store results
        pred_probs = np.empty((0,10), dtype= np.float32)
        dl_labels = np.empty((0,2), dtype= np.int32)   
                
        for idx, (x,y) in tqdm(enumerate(dataloader_ood), total= len(dataloader_ood)):
            
            x = x.cpu().numpy().squeeze(axis=1)
            y = y.cpu().numpy()

            probs = classifier.model.predict(x, verbose=0)
               
            pred_probs = np.append(pred_probs, probs, axis= 0)
            dl_labels = np.append(dl_labels, y, axis= 0)

        # from now on same custom code present in the analyzer function
        
        id_labels  =  dl_labels[:,0]                # filter by label column
        ood_labels =  dl_labels[:,1]
        prob_id     = pred_probs[id_labels == 1]    # split probabilites between ID adn OOD,
        prob_ood    = pred_probs[ood_labels == 1]
        
        # compute confidence (all)
        conf_all = round(self.compute_avgMSP(pred_probs),3)
        print("Confidence ID+OOD\t{}".format(conf_all))
        
        maximum_prob_id,  entropy_id,  mean_e_id,  std_e_id  = self.compute_statsProb(prob_id)
        maximum_prob_ood, entropy_ood, mean_e_ood, std_e_ood = self.compute_statsProb(prob_ood)
        
        # moments maximum prob 
        max_prob_id_mean    = np.mean(maximum_prob_id);     max_prob_id_std     = np.std(maximum_prob_id)
        max_prob_ood_mean   = np.mean(maximum_prob_ood);    max_prob_ood_std    = np.std(maximum_prob_ood)
        
        
        # in-out of distribution moments
        print("In-Distribution max prob         [mean (confidence ID),std]  -> ", max_prob_id_mean, max_prob_id_std)
        print("Out-Of-Distribution max prob     [mean (confidence OOD),std] -> ", max_prob_ood_mean, max_prob_ood_std)
        
        print("In-Distribution Entropy          [mean,std]                  -> ", mean_e_id, std_e_id)
        print("Out-Of-Distribution Entropy      [mean,std]                  -> ", mean_e_ood, std_e_ood)
        
        # normality detection
        print("Normality detection:")
        norm_base_rate = round(100*(prob_id.shape[0]/(prob_id.shape[0] + prob_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(norm_base_rate))
        print("\tKL divergence (entropy)")
        kl_norm_aupr, kl_norm_auroc = self.compute_curves(entropy_id, entropy_ood)
        print("\tPrediction probability")
        p_norm_aupr, p_norm_auroc = self.compute_curves(maximum_prob_id, maximum_prob_ood)
    
        # abnormality detection
        print("Abnormality detection:")
        abnorm_base_rate = round(100*(prob_ood.shape[0]/(prob_id.shape[0] + prob_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(abnorm_base_rate))
        print("\tKL divergence (entropy)")
        kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(-entropy_id, -entropy_ood, positive_reversed= True)
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-entropy_id, 1-entropy_ood, positive_reversed= True)
        print("\tPrediction probability")
        p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(-maximum_prob_id, -maximum_prob_ood, positive_reversed= True)
        # p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-maximum_prob_id, 1-maximum_prob_ood, positive_reversed= True)
    
        # compute fpr95, detection_error and threshold_error 
        metrics_norm = self.compute_metrics_ood(maximum_prob_id, maximum_prob_ood)
        print("OOD metrics:\n", metrics_norm)  

        # store statistics/metrics in a dictionary
        data = {
            "ID_max_prob": {
                "mean":  float(max_prob_id_mean), 
                "var":   float(max_prob_id_std)
            },
            "OOD_max_prob": {
                "mean": float(max_prob_ood_mean), 
                "var":  float(max_prob_ood_std) 
            },
            "ID_entropy":   {
                "mean": float(mean_e_id), 
                "var":  float(std_e_id)
            },
            "OOD_entropy":  {
                "mean": float(mean_e_ood), 
                "var":  float(std_e_ood)
            },
            
            "normality": {
                "base_rate":        float(norm_base_rate),
                "KL_AUPR":          float(kl_norm_aupr),
                "KL_AUROC":         float(kl_norm_auroc),
                "Prob_AUPR":        float(p_norm_aupr),   
                "Prob_AUROC":       float(p_norm_auroc)
            },
            "abnormality":{
                "base_rate":        float(abnorm_base_rate),
                "KL_AUPR":          float(kl_abnorm_aupr),
                "KL_AUROC":         float(kl_abnorm_auroc),
                "Prob_AUPR":        float(p_abnorm_aupr),   
                "Prob_AUROC":       float(p_abnorm_auroc),

            },
            "avg_confidence":   float(conf_all),
            "fpr95":            float(metrics_norm['fpr95']),
            "detection_error":  float(metrics_norm['detection_error']),
            "threshold":        float(metrics_norm['thr_de'])
            
        }
        
        print_dict(data)
        
        # save results
        path_model_test_results = "./results/ood_detection/model_test"
        check_folder(path_model_test_results)
        name_file = "baseline_simulated_experiment.json"
        saveJson(path_file = os.path.join(path_model_test_results, name_file), data = data)
        
    def test_threshold(self, thr_type = "fpr95_normality", normality_setting = True):
        """
            This function compute metrics (binary classification ID/OOD) that are threshold related (discriminator)
            
                Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset used as ood data, is is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            thr_type (str): choose which kind of threhsold use between "fpr95_normality" or "fpr95_abnormality" (fpr at tpr 95%) or "avg_confidence",
            "threshold_normality" or "threshold_abnormality",  Default is "fpr95_normality".
            normality setting" (str, optional): used to define positive label, in normality is ID data, in abnormality is OOD data. Default is True
            
        """
        
        # load data from analyze
        path_results_folder         = self.get_path2SaveResults()
        name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
        path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
        
        
        try:
            data = loadJson(path_result_save)
        except Exception as e:
            print(e)
            print("No data found at path {}".format(path_result_save))
        
        # choose the threshold to use for the discrimination ID/OOD
        if thr_type == "fpr95_normality":
            threshold = data['fpr95_normality']         # normality, positive label ID 
        elif thr_type == "threshold_normality":
            threshold = data['threshold_normality']
            
        elif thr_type == "fpr95_abnormality":
            threshold = data['fpr95_abnormality']       # abnormality, positive label OOD 
        elif thr_type == "threshold_abnormality":
            threshold = data['threshold_abnormality']
            
        else:   # for the normality setting                                      
            threshold = data['avg_confidence']          # normality, use avg max prob confidence as thr (proven to be misleading)
        
        id_ood_dataloader   = DataLoader(self.dataset_test,  batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # use model on selected device
        self.classifier.model.to(self.device)
        
        # define empty lsit to store outcomes
        test_logits     = np.empty((0,2), dtype= np.float32)
        test_labels     = np.empty((0,2), dtype= np.int32)
        
        for idx, (x,y) in tqdm(enumerate(id_ood_dataloader), total= len(id_ood_dataloader)):
            # to test
            # if idx >= 5: break
            
            x = x.to(self.device)
            with T.no_grad():
                with autocast():
                    _ ,_, logits =self.classifier.forward(x)
                    
            # to numpy array
            logits  = logits.cpu().numpy()
            y       = y.numpy()
                
            test_logits = np.append(test_logits, logits, axis= 0)
            test_labels = np.append(test_labels, y, axis= 0)
        
        probs = self._softmax(test_logits)
        
        print(probs.shape)
        maximum_prob    = np.max(probs, axis=1)
        print(maximum_prob.shape)
        
        pred = []
        for prob in maximum_prob:
            if prob < threshold: pred.append(0)     
            else: pred.append(1)                   
        
        # get the list with the binary labels
        pred = np.array(pred)
        
        if normality_setting:
            target = test_labels[:,0]   # if normal_setting, positive label is ID
        else:
            target = test_labels[:,1]    # if normal_setting, positive label is OOD 
        # compute and save metrics 
        name_resultClass_file  = 'metrics_ood_classification_{}.json'.format(self.name_ood_data)
        metrics_class =  metrics_binClass(preds = pred, targets= target, pred_probs = None, path_save = path_results_folder, name_ood_file = name_resultClass_file)
        
        print(metrics_class)

    def forward(self, x, thr_type = "fpr95_normality", normality_setting = True):
        """ discriminator forward

        Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset/CDDB scenario used as ood data, if is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            thr_type (str): choose which kind of threhsold use between "fpr95_normality" (fpr at tpr 95%) or "avg_confidence",
            "threshold_normality",  Default is "fpr95_normality".
            normality setting" (str, optional): used to define positive label, in normality is ID data, in abnormality is OOD data. Default is True
        """        
        if not(isinstance(x, T.Tensor)):
            x = T.tensor(x)
            
        # adjust to handle single image, increasing dimensions for batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        
        path_results_folder         = self.get_path2SaveResults(self)
        name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
        path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
        
        
        # load the threshold
        try:
            data = loadJson(path_result_save)
        except Exception as e:
            print(e)
            print("No data found at path {}".format(path_result_save))
            
        if thr_type == "fpr95_normality":
            threshold = data['fpr95_normality']         # normality, positive label ID 
        elif thr_type == "threshold_normality":
            threshold = data['threshold_normality']
            
        # elif thr_type == "fpr95_abnormality":
        #     threshold = data['fpr95_abnormality']       # abnormality, positive label OOD 
        # elif thr_type == "threshold_abnormality":
        #     threshold = data['threshold_abnormality']
            
        else:   # for the normality setting                                      
            threshold = data['avg_confidence']          # normality, use avg max prob confidence as thr (proven to be misleading)
        
        # compute mx prob 
        x = x.to(self.device)
        with T.no_grad():
             with autocast():
                _ ,_, logits =self.classifier.forward(x)
                
        # to numpy array
        logits  = logits.cpu().numpy()
        probs = self._softmax(logits)
        maximum_prob = np.max(probs, axis=1)
        
        # apply binary threshold
        if normality_setting:
            pred = np.where(condition= maximum_prob < threshold, x=0, y=1)  # if true set x otherwise set y
        else:
            pred = np.where(condition= maximum_prob < threshold, x=1, y=0)  # if true set x otherwise set y
        return pred

class Baseline_ODIN(OOD_Classifier):        # No model training necessary (Empty model forlder)
    """
        OOD detection baseline using softmax probability + ODIN framework
        
        IDEA: an higher MSP should be related to In-Distribution samples, a lower one instead to Out-Of-Distribution
        So values approching -> 1, are more likely to be ID, and the opposite with values approching 0.5
        are used perturbation and temperature to improve the separability between ID and OOD.
        
    """
    def __init__(self, classifier, task_type_prog, name_ood_data, id_data_test , ood_data_test , id_data_train = None, ood_data_train = None, useGPU = True):
        """
        OOD_Baseline instantiation 

        Args:
            classifier (DFD_BinClassifier): classifier used for the main Deepfake detection task
            ood_data_test (torch.utils.data.Dataset): test set out of distribution
            ood_data_train (torch.utils.data.Dataset): train set out of distribution
            useGPU (bool, optional): flag to enable usage of GPU. Defaults to True.
        """
        super(Baseline_ODIN, self).__init__(id_data_test = id_data_test, ood_data_test = ood_data_test,                  \
                                           id_data_train = id_data_train, ood_data_train = ood_data_train, useGPU = useGPU)
        # name of the OOD detection technique
        self.name = "ODIN+baseline"
        
        # set the classifier
        self.classifier         = classifier
        self.name_classifier    = self.classifier.classifier_name
        self.task_type_prog     = task_type_prog
        self.name_ood_data      = name_ood_data
        # load the Pytorch dataset here
        try:
            self.dataset_test  = OOD_dataset(self.id_data_test, self.ood_data_test, balancing_mode = "max")
        except:
            print("Dataset data is not valid, please use instances of class torch.Dataset")
        

        
    def test_probabilties(self):
        """ testing function using probabilty-based metrics, computing OOD metrics that are not threshold related
        """
        
        # saving folder path
        path_results_folder         = self.get_path2SaveResults()
        
        # define the dataloader 
        id_ood_dataloader = DataLoader(self.dataset_test,  batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # use model on selected device
        self.classifier.model.to(self.device)
        
        # define empty list to store outcomes
        pred_logits = np.empty((0,2), dtype= np.float32)
        dl_labels = np.empty((0,2), dtype= np.int32)            # dataloader labels, binary one-hot encoding, ID -> [1,0], OOD -> [0,1]
        
        # set temperature hyperparam
        odin_temperature = 1000
        odin_epsilon     = 0.00012  # 0.05
        
        for idx, (x,y) in tqdm(enumerate(id_ood_dataloader), total= len(id_ood_dataloader)):
            
            # to test
            # if idx >= 5: break
            
            x = x.to(self.device)
            y = y.to(self.device).to(T.float32)
            
            x_perturbed = self.odin_perturbations(x, self.classifier, is_batch=True, t = odin_temperature, epsilon= odin_epsilon)  # ODIN step 1
            
            with T.no_grad():
                with autocast():
                    _ ,_, logits =self.classifier.forward(x_perturbed)
              
            logits = logits/odin_temperature    # ODIN step 2
            
            # to numpy array
            logits  = logits.cpu().numpy()
            y       = y.detach().cpu().numpy()
            
            logits = logits - np.max(logits) # ODIN step 3
                
            pred_logits = np.append(pred_logits, logits, axis= 0)
            dl_labels = np.append(dl_labels, y, axis= 0)
            
        # to softmax probabilities
        probs = self._softmax(pred_logits)
   
        # separation of id/ood labels and probabilities
        id_labels  =  dl_labels[:,0]                # filter by label column
        ood_labels =  dl_labels[:,1]
        prob_id     = probs[id_labels == 1]         # split forward probabilites between ID adn OOD, still a list of probabilities for each class learned by the model
        prob_ood    = probs[ood_labels == 1]
        

        # compute confidence (all)
        conf_all = round(self.compute_avgMSP(probs),3)
        print("Confidence ID+OOD\t{}".format(conf_all))
        
        maximum_prob_id,  entropy_id,  mean_e_id,  std_e_id  = self.compute_statsProb(prob_id)
        maximum_prob_ood, entropy_ood, mean_e_ood, std_e_ood = self.compute_statsProb(prob_ood)
        
        max_prob_id_mean    = np.mean(maximum_prob_id);     max_prob_id_std     = np.std(maximum_prob_id)
        max_prob_ood_mean   = np.mean(maximum_prob_ood);    max_prob_ood_std    = np.std(maximum_prob_ood)
        
        
        # in-out of distribution moments
        print("In-Distribution max prob         [mean (confidence ID),std]  -> ", max_prob_id_mean, max_prob_id_std)
        print("Out-Of-Distribution max prob     [mean (confidence OOD),std] -> ", max_prob_ood_mean, max_prob_ood_std)
        
        print("In-Distribution Entropy          [mean,std]                  -> ", mean_e_id, std_e_id)
        print("Out-Of-Distribution Entropy      [mean,std]                  -> ", mean_e_ood, std_e_ood)
        
        # normality detection
        print("Normality detection:")
        norm_base_rate = round(100*(prob_id.shape[0]/(prob_id.shape[0] + prob_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(norm_base_rate))
        print("\tKL divergence (entropy)")
        kl_norm_aupr, kl_norm_auroc = self.compute_curves(entropy_id, entropy_ood)
        print("\tPrediction probability")
        p_norm_aupr, p_norm_auroc = self.compute_curves(maximum_prob_id, maximum_prob_ood)

        
        # abnormality detection
        print("Abnormality detection:")
        abnorm_base_rate = round(100*(prob_ood.shape[0]/(prob_id.shape[0] + prob_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(abnorm_base_rate))
        print("\tKL divergence (entropy)")
        kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-entropy_id, 1-entropy_ood, positive_reversed= True)
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-entropy_id, 1-entropy_ood, positive_reversed= True)
        print("\tPrediction probability")
        p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-maximum_prob_id, 1-maximum_prob_ood, positive_reversed= True)
        # p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-maximum_prob_id, 1-maximum_prob_ood, positive_reversed= True)

        # compute fpr95, detection_error and threshold_error 
        metrics_norm = self.compute_metrics_ood(maximum_prob_id, maximum_prob_ood, path_save = path_results_folder)
        print("OOD metrics:\n", metrics_norm)  
        metrics_abnorm = self.compute_metrics_ood(1-maximum_prob_id, 1-maximum_prob_ood, positive_reversed= True)
        print("OOD metrics:\n", metrics_abnorm)   
                     
        # store statistics/metrics in a dictionary
        data = {
            "ID_max_prob": {
                "mean":  float(max_prob_id_mean), 
                "var":   float(max_prob_id_std)
            },
            "OOD_max_prob": {
                "mean": float(max_prob_ood_mean), 
                "var":  float(max_prob_ood_std) 
            },
            "ID_entropy":   {
                "mean": float(mean_e_id), 
                "var":  float(std_e_id)
            },
            "OOD_entropy":  {
                "mean": float(mean_e_ood), 
                "var":  float(std_e_ood)
            },
            
            "normality": {
                "base_rate":        float(norm_base_rate),
                "KL_AUPR":          float(kl_norm_aupr),
                "KL_AUROC":         float(kl_norm_auroc),
                "Prob_AUPR":        float(p_norm_aupr),   
                "Prob_AUROC":       float(p_norm_auroc)
            },
            "abnormality":{
                "base_rate":        float(abnorm_base_rate),
                "KL_AUPR":          float(kl_abnorm_aupr),
                "KL_AUROC":         float(kl_abnorm_auroc),
                "Prob_AUPR":        float(p_abnorm_aupr),   
                "Prob_AUROC":       float(p_abnorm_auroc),

            },
            "avg_confidence":   float(conf_all),
            "fpr95_normality":              float(metrics_norm['fpr95']),
            "detection_error_normality":    float(metrics_norm['detection_error']),
            "threshold_normality":          float(metrics_norm['thr_de']),
            "fpr95_abnormality":            float(metrics_abnorm['fpr95']),
            "detection_error_abnormality":  float(metrics_abnorm['detection_error']),
            "threshold_abnormality":        float(metrics_abnorm['thr_de'])
            
        }
        
        # save data (JSON)
        if self.name_ood_data is not None:
            name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
            path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
            
            print(path_result_save)
                  
            saveJson(path_file = path_result_save, data = data)
        
    def forward(self, x, thr_type = "fpr95_normality", normality_setting = True):
        """ discriminator forward

        Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset/CDDB scenario used as ood data, if is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            thr_type (str): choose which kind of threhsold use between "fpr95" (fpr at tpr 95%) or "avg_confidence", or "thr_de",  Default is "fpr95".
            thr_type (str): choose which kind of threhsold use between "fpr95_normality" (fpr at tpr 95%) or "avg_confidence",
                            "threshold_normality",  Default is "fpr95_normality".
            normality setting" (str, optional): used to define positive label, in normality is ID data, in abnormality is OOD data. Default is True
        """        
        if not(isinstance(x, T.Tensor)):
            x = T.tensor(x)
            
        # adjust to handle single image, increasing dimensions for batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        
        path_results_folder         = self.get_path2SaveResults(self)
        name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
        path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
        
        
        # load the threshold
        try:
            data = loadJson(path_result_save)
        except Exception as e:
            print(e)
            print("No data found at path {}".format(path_result_save))
            
        if thr_type == "fpr95_normality":
            threshold = data['fpr95_normality']         # normality, positive label ID 
        elif thr_type == "threshold_normality":
            threshold = data['threshold_normality']
            
        else:   # for the normality setting                                      
            threshold = data['avg_confidence']          # normality, use avg max prob confidence as thr (proven to be misleading)
        
        # compute mx prob 
        x = x.to(self.device)
        x = self.odin_perturbations(x,self.classifier, is_batch=True)
        
        with T.no_grad():
             with autocast():
                _ ,_, logits =self.classifier.forward(x)
                
        # to numpy array
        logits  = logits.cpu().numpy()
        probs = self.softmax_temperature(logits)
        maximum_prob = np.max(probs, axis=1)
        
        # apply binary threshold
        if normality_setting:
            pred = np.where(condition= maximum_prob < threshold, x=0, y=1)  # if true set x otherwise set y
        else:
            pred = np.where(condition= maximum_prob < threshold, x=1, y=0)  # if true set x otherwise set y
        return pred

class Confidence_Detector(OOD_Classifier):
    """
        OOD detection using confidence estimation from the model
        
        *** Required confidence computation ***
        
    """
    def __init__(self, classifier,  task_type_prog, name_ood_data, id_data_test , ood_data_test , id_data_train = None, ood_data_train = None, useGPU = True):
        """
        OOD_Baseline instantiation 

        Args:
            classifier (DFD_BinClassifier): classifier used for the main Deepfake detection task, required confidence computation for the model 
            ood_data_test (torch.utils.data.Dataset): test set out of distribution
            ood_data_train (torch.utils.data.Dataset): train set out of distribution
            useGPU (bool, optional): flag to enable usage of GPU. Defaults to True.
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset used as ood data, is is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
                if None, the results will be not saved
        """
        super(Confidence_Detector, self).__init__(id_data_test = id_data_test, ood_data_test = ood_data_test,                  \
                                           id_data_train = id_data_train, ood_data_train = ood_data_train, useGPU = useGPU)
        # set the classifier
        self.classifier  = classifier
        self.name_classifier = self.classifier.classifier_name
        self.check_classifier()
        
        self.task_type_prog = task_type_prog
        self.name_ood_data  = name_ood_data
        
        # load the Pytorch dataset here
        try:
            self.dataset_test  = OOD_dataset(self.id_data_test, self.ood_data_test, balancing_mode = "max")
        except:
            print("Dataset data is not valid, please use instances of class torch.Dataset")
        
        # name of the classifier
        self.name = "confidenceDetector"
    
    
    def check_classifier(self):
        try:
            assert "confidence" in self.classifier.model_type.lower().strip()
        except Exception as e:
            raise ValueError("The classifier model does not provide a confidence estimation")  
    #                                       testing functions
    def test_confidence(self):
        """
            testing function using probabilty-based metrics, computing OOD metrics that are not threshold related
        """
        
        # saving folder path
        path_results_folder         = self.get_path2SaveResults()
        
        # define the dataloader 
        id_ood_dataloader   = DataLoader(self.dataset_test,  batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # use model on selected device
        self.classifier.model.to(self.device)
        
        # define empty list to store outcomes
        confidences = np.empty((0,1), dtype= np.float32)
        dl_labels = np.empty((0,2), dtype= np.int32)            # dataloader labels, binary one-hot encoding, ID -> [1,0], OOD -> [0,1]
        
        for idx, (x,y) in tqdm(enumerate(id_ood_dataloader), total= len(id_ood_dataloader)):
            
            # to test
            # if idx >= 10: break
            
            x = x.to(self.device)
            with T.no_grad():
                with autocast():
                    # _ ,_, logits =self.classifier.forward(x)
                    out = self.classifier.model.forward(x)
                    confidence = out[3]
                    
            # to numpy array
            confidence  = confidence.cpu().numpy()
            y       = y.numpy()
                
            confidences = np.append(confidences, confidence, axis= 0)
            dl_labels = np.append(dl_labels, y, axis= 0)
            

        # separation of id/ood labels and probabilities
        id_labels  =  dl_labels[:,0]                # filter by label column
        ood_labels =  dl_labels[:,1]
        confidences_id     = confidences[id_labels == 1]         # split forward probabilites between ID adn OOD, still a list of probabilities for each class learned by the model
        confidences_ood    = confidences[ood_labels == 1]
        

        # compute confidence (all)
        conf_all = round(np.average(confidences),3)
        print("Confidence ID+OOD\t{}".format(conf_all))
        
        conf_id_mean    = np.mean(confidences_id);     conf_id_std     = np.std(confidences_id)
        conf_ood_mean   = np.mean(confidences_ood);    conf_ood_std    = np.std(confidences_ood)
        
        
        # in-out of distribution moments
        print("In-Distribution confidence        [mean (confidence ID),std]  -> ", conf_id_mean, conf_id_std)
        print("Out-Of-Distribution confidence     [mean (confidence OOD),std] -> ", conf_ood_mean, conf_ood_std)
        
        # normality detection
        print("Normality detection:")   # positive label -> ID data
        norm_base_rate = round(100*(confidences_id.shape[0]/(confidences_id.shape[0] + confidences_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(norm_base_rate))
        print("\tPrediction confidence")
        p_norm_aupr, p_norm_auroc = self.compute_curves(confidences_id, confidences_ood)

        
        # abnormality detection
        print("Abnormality detection:")   # positive label -> OOD data
        abnorm_base_rate = round(100*(confidences_ood.shape[0]/(confidences_id.shape[0] + confidences_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(abnorm_base_rate))
        print("\tPrediction confidence")
        p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-confidences_id, 1-confidences_ood, positive_reversed= True)
        # p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-maximum_prob_id, 1-maximum_prob_ood, positive_reversed= True)

        # compute fpr95, detection_error and threshold_error 
        metrics_norm = self.compute_metrics_ood(confidences_id, confidences_ood, path_save = path_results_folder)
        print("OOD metrics:\n", metrics_norm)  
        metrics_abnorm = self.compute_metrics_ood(1-confidences_id, 1-confidences_ood, positive_reversed= True)
        print("OOD metrics:\n", metrics_abnorm)   
                     
        # store statistics/metrics in a dictionary
        data = {
            "ID_confidence": {
                "mean":  float(conf_id_mean), 
                "var":   float(conf_id_std)
            },
            "OOD_confidence": {
                "mean": float(conf_ood_mean), 
                "var":  float(conf_ood_std) 
            },
            
            "normality": {
                "base_rate":                float(norm_base_rate),
                "Confidence_AUPR":          float(p_norm_aupr),   
                "Confidence_AUROC":         float(p_norm_auroc)
            },
            "abnormality":{
                "base_rate":                float(abnorm_base_rate),
                "Confidence_AUPR":          float(p_abnorm_aupr),   
                "Confidence_AUROC":         float(p_abnorm_auroc),

            },
            "avg_confidence":               float(conf_all),
            "fpr95_normality":              float(metrics_norm['fpr95']),
            "detection_error_normality":    float(metrics_norm['detection_error']),
            "threshold_normality":          float(metrics_norm['thr_de']),
            "fpr95_abnormality":            float(metrics_abnorm['fpr95']),
            "detection_error_abnormality":  float(metrics_abnorm['detection_error']),
            "threshold_abnormality":        float(metrics_abnorm['thr_de'])
              
        }
        
        # save data (JSON)
        if self.name_ood_data is not None:
            path_results_folder         = self.get_path2SaveResults()
            name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
            path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
            
            print(path_result_save)
                  
            saveJson(path_file = path_result_save, data = data)
    
    def test_threshold(self, thr_type = "fpr95_normality", normality_setting = True):
        """
            This function compute metrics (binary classification ID/OOD) that are threshold related (discriminator)
            
                Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset used as ood data, is is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            thr_type (str): choose which kind of threhsold use between "fpr95_normality" or "fpr95_abnormality" (fpr at tpr 95%) or "avg_confidence",
            "threshold_normality" or "threshold_abnormality",  Default is "fpr95_normality".
            normality setting" (str, optional): used to define positive label, in normality is ID data, in abnormality is OOD data. Default is True
            
        """
        
        # load data from analyze
        path_results_folder         = self.get_path2SaveResults()
        name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
        path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
        
        
        try:
            data = loadJson(path_result_save)
        except Exception as e:
            print(e)
            print("No data found at path {}".format(path_result_save))
        
        # choose the threshold to use for the discrimination ID/OOD
        if thr_type == "fpr95_normality":
            threshold = data['fpr95_normality']         # normality, positive label ID 
        elif thr_type == "threshold_normality":
            threshold = data['threshold_normality']
            
        elif thr_type == "fpr95_abnormality":
            threshold = data['fpr95_abnormality']       # abnormality, positive label OOD 
        elif thr_type == "threshold_abnormality":
            threshold = data['threshold_abnormality']
            
        else:   # for the normality setting                                      
            threshold = data['avg_confidence']          # normality, use avg max prob confidence as thr (proven to be misleading)
        
        id_ood_dataloader   = DataLoader(self.dataset_test,  batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # use model on selected device
        self.classifier.model.to(self.device)
        
        # define empty lsit to store outcomes
        test_confidences = np.empty((0,1), dtype= np.float32)
        test_labels     = np.empty((0,2), dtype= np.int32)
        
        for idx, (x,y) in tqdm(enumerate(id_ood_dataloader), total= len(id_ood_dataloader)):
            # to test
            # if idx >= 5: break
            
            x = x.to(self.device)
            with T.no_grad():
                with autocast():
                    out = self.classifier.model.forward(x)
                    confidence = out[3]
                    
            # to numpy array
            confidence  = confidence.cpu().numpy()
            y       = y.numpy()
                
            test_confidences = np.append(test_confidences, confidence, axis= 0)
            test_labels = np.append(test_labels, y, axis= 0)
        
        
        test_confidences = np.squeeze(test_confidences) 
        pred = []
        for prob in test_confidences:
            if prob < threshold: pred.append(0)     
            else: pred.append(1)                   
        
        # get the list with the binary labels
        pred = np.array(pred)
        
        if normality_setting:
            target = test_labels[:,0]   # if normal_setting, positive label is ID
        else:
            target = test_labels[:,1]    # if normal_setting, positive label is OOD 
        # compute and save metrics 
        name_resultClass_file  = 'metrics_ood_classification_{}.json'.format(self.name_ood_data)
        metrics_class =  metrics_binClass(preds = pred, targets= target, pred_probs = None, path_save = path_results_folder, name_ood_file = name_resultClass_file)
        
        print(metrics_class)

    def forward(self, x, thr_type = "fpr95_normality", normality_setting = True):
        """ discriminator forward

        Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset/CDDB scenario used as ood data, if is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            thr_type (str): choose which kind of threhsold use between "fpr95_normality" (fpr at tpr 95%) or "avg_confidence",
            "threshold_normality",  Default is "fpr95_normality".
            normality setting" (str, optional): used to define positive label, in normality is ID data, in abnormality is OOD data. Default is True
        """        
        if not(isinstance(x, T.Tensor)):
            x = T.tensor(x)
            
        # adjust to handle single image, increasing dimensions for batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        
        path_results_folder         = self.get_path2SaveResults(self)
        name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
        path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
        
        
        # load the threshold
        try:
            data = loadJson(path_result_save)
        except Exception as e:
            print(e)
            print("No data found at path {}".format(path_result_save))
            
        if thr_type == "fpr95_normality":
            threshold = data['fpr95_normality']         # normality, positive label ID 
        elif thr_type == "threshold_normality":
            threshold = data['threshold_normality']
            
        # elif thr_type == "fpr95_abnormality":
        #     threshold = data['fpr95_abnormality']       # abnormality, positive label OOD 
        # elif thr_type == "threshold_abnormality":
        #     threshold = data['threshold_abnormality']
            
        else:   # for the normality setting                                      
            threshold = data['avg_confidence']          # normality, use avg max prob confidence as thr (proven to be misleading)
        
        # compute mx prob 
        x = x.to(self.device)
        with T.no_grad():
             with autocast():
                    out = self.classifier.model.forward(x)
                    confidence = out[3]
                    
        # to numpy array
        confidence  = confidence.cpu().numpy()
        confidence  = np.squeeze(confidence) 
        
        # apply binary threshold
        if normality_setting:
            pred = np.where(condition= confidence < threshold, x=0, y=1)  # if true set x otherwise set y
        else:
            pred = np.where(condition= confidence < threshold, x=1, y=0)  # if true set x otherwise set y
        return pred
    
class Abnormality_module(OOD_Classifier):   # model to train necessary 
    """ Custom implementation of the abnormality module using Unet4, look https://arxiv.org/abs/1610.02136 chapter 4"
    
    model_type (str): choose between: "basic", 
    
    IDEA: define a custom model that uses several information from the classifier (softmax probs, encoding, residual) to
    generate a risk score (from 0 to 1) that represents the likely of a sample to be Out of Distribution.
    So values approching -> 1, are more likely to be OOD, and the opposite with values approching 0.5 or lower
    
    """
    
    def __init__(self, classifier, scenario:str, model_type: str, useGPU: bool= True, binary_dataset: bool = True,
                 batch_size = "dafault", use_synthetic:bool = True, extended_ood: bool = False, blind_test: bool = True,
                 balancing_mode: str = "max", conf_usage_mode: str = "merge"):
        """ 
            ARGS:
            - classifier (T.nn.Module): the classifier (Module A) that produces the input for Module B (abnormality module)
            - scenario (str): choose between: "content", "mix", "group"
            - model_type (str): choose between avaialbe model for the abnrormality module: "basic", "encoder", "encoder_v2", "encoder_v3"
            "encoder_v4"
            - batch_size (str/int): the size of the batch, set defaut to use the assigned from superclass, otherwise the int size. Default is "default".
            - use_synthetic (boolean): choose if use ood data generated from ID data (synthetic) with several techniques, or not. Defaults is True.
            - extended_ood (boolean, optional): This has sense if use_synthetic is set to True. Select if extend the ood data for training, using not only synthetic data. Default is True
            - blind_test (boolean, optional): This has sense if use_synthetic is set to True. Select if use real ood data (True) or synthetized one from In distributiion data. Default is True
            - balancing_mode (string,optinal): This has sense if use_synthethid is set to True and extended_ood is set to True.
            Choose between "max and "all", max mode give a balance number of OOD same as ID, while, all produces more OOD samples than ID. Default is "max"
            - conf_usage_mode (string, optional): This has sense only if model include confidence inference. Choose how use the confidence model. "merge" mode
            combine (stack) the confidence with the probabilities vector, "ignore" avoid the use of confidence for the ood detection, "alone" exchange the probability
            vector with the confidence degree.
            
        """
        super(Abnormality_module, self).__init__(useGPU=useGPU)
        
        # set the classifier (module A)
        self.classifier  = classifier
        self.scenario = scenario
        self.binary_dataset = binary_dataset
        self.model_type = model_type
        
        self.name               = "Abnormality_module"
        self.name_classifier    = self.classifier.classifier_name
        self.train_name         = None
        self._meta_data()
        
        # configuration variables for abnormality module (module B)
        self.use_confidence     = False            # as default setting, the confidence is not expected, this is modified by _build_model() if confidence is present
        self.augment_data_train = False
        self.loss_name          = "weighted bce"   # binary cross entropy or sigmoid cross entropy (weighted)
        self.conf_usage_mode    = conf_usage_mode  # define how use confidence whether is present
        
        # instantiation aux elements
        self.bce     = F.binary_cross_entropy_with_logits   # performs sigmoid internally
        # self.ce      = F.cross_entropy()
        self.sigmoid = F.sigmoid
        self.softmax = F.softmax
        self._build_model()
            
        # training parameters  
        if not batch_size == "dafault":   # default batch size is defined in the superclass 
            self.batch_size             = int(batch_size)
            
        # self.lr                     = 1e-4
        self.lr                     = 1e-3
        self.n_epochs               = 50
        self.weight_decay           = 1e-3                  # L2 regularization term 
        
        # load data ID/OOD
        if self.binary_dataset:   # binary vs multi-class task
            self.dataset_class = CDDB_binary_Partial
        else:
            self.dataset_class = CDDB_Partial
        
        # Datasets flags
        self.use_synthetic  = use_synthetic    
        self.extended_ood   = extended_ood
        self.blind_test     = blind_test
        if not(balancing_mode in ["max", "all"]):
            raise ValueError('Wrong selection for balancing mode. Choose between "max" or "all".') 
        self.balancing_mode = balancing_mode
        
        # Define sets
        if self.use_synthetic:
            self._prepare_data_synt(verbose = True)
        else:
            self._prepare_data(verbose=True)
        

           
    def _build_model(self):
        
        # check if the model estimate the confidence and if should be not ignored 
        if ("confidence" in self.name_classifier.lower().strip()) and (self.conf_usage_mode.lower().strip() != "ignore") :
            self.use_confidence = True 
        
        # select the type of encoding, encoder_v3 is a smaller dimensionality
        if self.model_type in ["basic", "encoder", "encoder_v2", "encoder_v4"]:
            self.classifier.model.large_encoding = True
        else:
            self.classifier.model.large_encoding = False
        
        # compute shapes for the input
        x_shape = (1, *self.classifier.model.input_shape)
        x = T.rand(x_shape).to(self.device)
        out = self._forward_A(x)
        

        probs_softmax       = out["probabilities"]
        encoding            = out["encoding"]
        residual_flatten    = out["residual"]
        
        
        # include confidence directly in the probs_softmax according to the mode choosen in self.conf_usage_mode (this in order to don't change the whole strcuture of abnorom module)
        if self.use_confidence:
            confidence = out["confidence"]
            if self.conf_usage_mode.lower().strip() == "merge":
                probs_softmax = T.cat((probs_softmax, confidence),dim = 1)
            elif self.conf_usage_mode.lower().strip() == "alone":
                probs_softmax = confidence
            else: 
                raise ValueError('invalid modality for the confidence usage, chosen: {}, valid are: "merge","alone","ignore"'.format(self.conf_usage_mode))
        
        
        if self.model_type == "basic":
            self.model = Abnormality_module_Basic(probs_softmax.shape, encoding.shape, residual_flatten.shape)
        if self.model_type == "encoder":
            self.model = Abnormality_module_Encoder_v1(probs_softmax.shape, encoding.shape, residual_flatten.shape)
        if self.model_type == "encoder_v2":
            self.model = Abnormality_module_Encoder_v2(probs_softmax.shape, encoding.shape, residual_flatten.shape)
        if self.model_type == "encoder_v3":
            self.model = Abnormality_module_Encoder_v3(probs_softmax.shape, encoding.shape, residual_flatten.shape)
        if self.model_type == "encoder_v4":  
            self.model = Abnormality_module_Encoder_v4(probs_softmax.shape, encoding.shape, residual_flatten.shape)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _meta_data(self, task_type_prog = None):
        """ prepare meta data of the current model: task_type_prog and name_ood_data"""
        
        if task_type_prog is None:
            if self.binary_dataset:
                task_type_prog = 0 # binary-class
            else:
                task_type_prog = 1  # multi-class 
        self.task_type_prog = task_type_prog 
        
        try:
            setting = getScenarioSetting()[self.scenario]
        except:
            ValueError("wrong selection for the scenario, is not possible to retrieve the setting")
        
        # if self.scenario == "content":
        #     self.name_ood_data  = "CDDB_" + self.scenario + "_faces_scenario"
        # else:
        #     self.name_ood_data  = "CDDB_" + self.scenario + "_scenario"
        
        self.name_ood_data  = "CDDB_" + self.scenario + "_" + setting + "_scenario"
        
    def _prepare_data_synt(self,verbose = False):
        """ method used to prepare Dataset class used for both training and testing, synthetizing OOD data for training
        
            ARGS:
            - verbose (boolean, optional): choose to print extra information while loading the data
        """
        
        # synthesis of OOD data (train and valid)
        print("\n\t\t[Loading OOD (synthetized) data]\n")
        ood_data_train_syn    = self.dataset_class(scenario = self.scenario, train = True,  ood = False, augment = False, transform2ood = True)
        tmp        = self.dataset_class(scenario = self.scenario, train = False, ood = False, augment = False, transform2ood = True)
        ood_data_valid_syn , ood_data_test_syn      = sampleValidSet(trainset = ood_data_train_syn, testset= tmp, useOnlyTest = True, verbose = True)
        
        # fetch ID data (train, valid and test)
        print("\n\t\t[Loading ID data]\n")
        id_data_train      = self.dataset_class(scenario = self.scenario, train = True,  ood = False, augment = False, transform2ood = False)
        tmp            = self.dataset_class(scenario = self.scenario, train = False, ood = False, augment = False, transform2ood = False)
        id_data_valid , id_data_test   = sampleValidSet(trainset = id_data_train, testset= tmp, useOnlyTest = True, verbose = True)
        
        
        if verbose:
            print("length ID dataset  (train) -> ",  len(id_data_train))
            print("length ID dataset  (valid) -> ",  len(id_data_valid))
            print("length ID dataset  (test) -> ", len(id_data_test))
            print("length OOD dataset (train) synthetized -> ", len(ood_data_train_syn))
            print("length OOD dataset (valid) synthetized -> ", len(ood_data_valid_syn))

                
        if self.extended_ood:
            print("\n\t\t[Extending OOD data with CDDB samples]\n")
            ood_train_expansion = self.dataset_class(scenario = self.scenario, train = True,   ood = True, augment = False)            
            ood_data_train = mergeDatasets(ood_data_train_syn, ood_train_expansion) 
           
            if verbose: print("length OOD dataset after extension (train) -> ", len(ood_data_train))
            
            # train set: id data train + ood from synthetic ood and expansion)
            self.dataset_train = OOD_dataset(id_data_train, ood_data_train, balancing_mode= self.balancing_mode)
        else:
            # train set: id data train + synthetic ood (id data train transformed in ood)
            self.dataset_train = OOD_dataset(id_data_train, ood_data_train_syn, balancing_mode= self.balancing_mode)
            
        if self.blind_test:
            ood_data_test  = self.dataset_class(scenario = self.scenario, train = False,  ood = True, augment = False)
            if verbose: print("length OOD dataset (test) -> ", len(ood_data_test))
            # test set: id data test + ood data test
            self.dataset_test  = OOD_dataset(id_data_test , ood_data_test,  balancing_mode= self.balancing_mode)
        else:
            self.dataset_test  = OOD_dataset(id_data_test , ood_data_test_syn,  balancing_mode= self.balancing_mode)  # not real ood data but the synthetized one (useful to test the effective learning of the model)
        
        # valid set: id data valid + synthetic ood (id data train transformed in ood)
        self.dataset_valid = OOD_dataset(id_data_valid, ood_data_valid_syn, balancing_mode= self.balancing_mode)
        
        
        if verbose: print("length full dataset (train/valid/test) with balancing -> ", len(self.dataset_train), len(self.dataset_valid), len(self.dataset_test))
        print("\n")
    
    def _prepare_data(self, verbose = False):
        
        """ method used to prepare Dataset class used for both training and testing, both ID and OOD comes from CDDB dataset
        
            ARGS:
           - verbose (boolean, optional): choose to print extra information while loading the data
        """
        
        # fetch ID data (train, valid and test)
        print("\n\t\t[Loading ID data]\n")
        id_data_train      = self.dataset_class(scenario = self.scenario, train = True,  ood = False, augment = False, transform2ood = False)
        tmp            = self.dataset_class(scenario = self.scenario, train = False, ood = False, augment = False, transform2ood = False)
        id_data_valid , id_data_test   = sampleValidSet(trainset = id_data_train, testset= tmp, useOnlyTest = True, verbose = True)
        

        print("\n\t\t[Loading OOD data]\n")
        ood_data_train = self.dataset_class(scenario = self.scenario, train = True,   ood = True, augment = False) 
        tmp  = self.dataset_class(scenario = self.scenario, train = False,  ood = True, augment = False)
        ood_data_valid , ood_data_test   = sampleValidSet(trainset = ood_data_train, testset= tmp, useOnlyTest = True, verbose = True)
        
        
        if verbose:
            print("length ID dataset  (train) -> ",  len(id_data_train))
            print("length ID dataset  (valid) -> ",  len(id_data_valid))
            print("length ID dataset  (test)  -> ", len(id_data_test))
        if verbose: 
            print("length OOD dataset (train) -> ", len(ood_data_train))
            print("length OOD dataset (valid) -> ", len(ood_data_valid))
            print("length OOD dataset (test)  -> ", len(ood_data_test))

                
        # define the OOD detection sets
        self.dataset_train = OOD_dataset(id_data_train, ood_data_train, balancing_mode=self.balancing_mode)
        self.dataset_test  = OOD_dataset(id_data_test , ood_data_valid, balancing_mode=self.balancing_mode)
        self.dataset_valid = OOD_dataset(id_data_valid, ood_data_test , balancing_mode=self.balancing_mode)
        
        if verbose: print("length full dataset (train/valid/test) with balancing -> ", len(self.dataset_train), len(self.dataset_valid), len(self.dataset_test))
        print("\n")
     
    def _hyperParams(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs_max": self.n_epochs,
            "weight_decay": self.weight_decay
                }
    
    def _dataConf(self):
        
        # load not fixed config specs with try-catch

        try:
            input_shape = str(self.model.input_shape)
        except:
            input_shape = "empty"
            
        try:
            large_encoding_classifier  = self.classifier.model.large_encoding
        except:
            large_encoding_classifier = True
        
        return {
            "date_training": date.today().strftime("%d-%m-%Y"),
            "model": "Abnormality module " + self.model_type,
            "large_encoding_classifier": large_encoding_classifier,
            "input_shape": input_shape,
            "use confidence": self.use_confidence,
            "confidence usage mode": self.conf_usage_mode,
            "data_scenario": self.scenario,
            "optimizer": self.optimizer.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__,
            "loss": self.loss_name,
            "grad_scaler": True,                # always true
            "base_augmentation": self.augment_data_train,
            "Use OOD data synthetized":  self.use_synthetic,
            "Use extension OOD from CDDB": self.extended_ood,
            "Balancing mode": self.balancing_mode,

            # dataset lengths 
            "Train Set Samples": len(self.dataset_train),
            "Valid Set Samples": len(self.dataset_valid),
            "Test Set Samples":  len(self.dataset_test),
            
            # dataset distribution
            "ID train samples": round((1/self.weights_labels[0]) * self.samples_train),
            "OOD train samples": round((1/self.weights_labels[1]) * self.samples_train)
            }
    
    def init_logger(self, path_model):
        """
            path_model -> specific path of the current model training
        """
        logger = ExpLogger(path_model=path_model)
        
        # logger for the classifier (module A)
        logger.write_config(self.classifier._dataConf(), name_section="Configuration Classifier")
        logger.write_hyper(self.classifier._hyperParams(), name_section="Hyperparameters Classifier")
        try:
            logger.write_model(self.classifier.model.getSummary(verbose=False), name_section="Model architecture Classifier")
        except:
            print("Impossible to retrieve the model structure for logging")
        
        # logger for the abnormality module  (module B)
        logger.write_config(self._dataConf())
        logger.write_hyper(self._hyperParams())
        try:
            logger.write_model(self.model.getSummary(verbose=False))
        except:
            print("Impossible to retrieve the model structure for logging")
        
        return logger
    
    def _forward_A(self, x, verbose = False):
        """ this method return a dictionary with the all the outputs from the model branches
            keys: "probabilities", "encding", "residual", "confidence"
            the confidence key-value pair is present if and only if is a confidence model (check the name)
        """

        # logits, reconstruction, encoding = self.classifier.model.forward(x)
        
        output_model = self.classifier.model.forward(x)
        
        # unpack the output based on the model
        logits          = output_model[0]
        reconstruction  = output_model[1]
        encoding        = output_model[2]
        
        output = {"encoding": encoding}
        
        if self.use_confidence:
            confidence = output_model[3]
            output["confidence"] = confidence
        
        probs_softmax = T.nn.functional.softmax(logits, dim=1)
        
        if verbose: 
            print("prob shape -> ", probs_softmax.shape)
            print("encoding shape -> ",encoding.shape)
        
        # from reconstuction to residual
        residual = T.square(reconstruction - x)
        # residual_flatten = T.flatten(residual, start_dim=1)
        
        output["probabilities"] = probs_softmax
        output["residual"]      = residual
        
        if verbose: 
            print("residual shape ->", reconstruction.shape)
        
        return output
                
        # if self.use_confidence:
        #     return probs_softmax, encoding, residual, confidence
        # else:
        #     return probs_softmax, encoding, residual

    def _forward_B(self, probs_softmax, encoding, residual, verbose = False):
        """ if self.use_confidence is True probs_softmax should include the confidence value (Stacked) """
        
        
        y = self.model.forward(probs_softmax, encoding, residual)
        if verbose: print("y shape", y.shape)
        return y
        
    def forward(self,x):
        
        if self.model is None: raise ValueError("No model has been defined, impossible forwarding the data")
        
        if not(isinstance(x, T.Tensor)):
            x = T.tensor(x)
        
        # handle single image, increasing dimensions simulating a batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        elif len(x.shape) <= 2 or len(x.shape) >= 5:
            raise ValueError("The input shape is not compatiple, expected a batch or a single image")
        
        # correct the dtype
        if not (x.dtype is T.float32):
            x = x.to(T.float32)
         
        x = x.to(self.device)
        
        out = self._forward_A(x)

        probs_softmax       = out["probabilities"]
        encoding            = out["encoding"]
        residual            = out["residual"]
        
        if self.use_confidence:
            confidence = out["confidence"]
            if self.conf_usage_mode.lower().strip() == "merge":
                probs_softmax = T.cat((probs_softmax, confidence),dim = 1)
            elif self.conf_usage_mode.lower().strip() == "alone":
                probs_softmax = confidence
        
        
        logit = self._forward_B(probs_softmax, encoding, residual)
        
        out   = self.sigmoid(logit)
        
        return  out
    
    def load(self, name_folder, epoch):
        
        print("\n\t\t[Loading model]\n")
        
        self._meta_data()
        
        # save folder of the train (can be used to save new files in models and results)
        self.train_name     = name_folder
        self.modelEpochs    = epoch
        
        # get full path to load
        models_path         = self.get_path2SaveModels()
        path2model          = os.path.join(models_path,  name_folder, str(epoch) + ".ckpt")

        try:
            loadModel(self.model, path2model)
            self.model.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except Exception as e:
            print(e)
            print("No model: {} found for the epoch: {} in the folder: {}".format(name_folder, epoch, path2model))
    
    def valid(self, epoch, valid_dl):
        """
            validation method used mainly for the Early stopping training
        """
        print (f"Validation for the epoch: {epoch} ...")
        
        # set temporary evaluation mode and empty cuda cache
        self.model.eval()
        T.cuda.empty_cache()
        
        # list of losses
        losses = []

        
        for (x,y) in tqdm(valid_dl):
            
            x = x.to(self.device)
            # y = y.to(self.device).to(T.float32)
            
            # take only label for the positive class (fake)
            y = y[:,1]
            
            # compute monodimensional weights for the full batch
            weights = T.tensor([self.weights_labels[elem] for elem in y ]).to(self.device)
            
            y = y.to(self.device).to(T.float32)

            
            with T.no_grad():
                with autocast():
                    
                    out = self._forward_A(x)

                    probs_softmax       = out["probabilities"]
                    encoding            = out["encoding"]
                    residual            = out["residual"]
                    
                    if self.use_confidence:
                        confidence = out["confidence"]
                        if self.conf_usage_mode.lower().strip() == "merge":
                            probs_softmax = T.cat((probs_softmax, confidence),dim = 1)
                        elif self.conf_usage_mode.lower().strip() == "alone":
                            probs_softmax = confidence

                    if self.model_type == "basic" or "encoder" in self.model_type:
                        # logit = self._forward_B(prob_softmax, encoding, residual)
                        # logit = T.squeeze(logit)
                        logit = T.squeeze(self.model.forward(probs_softmax, encoding, residual))
                    else:
                        raise ValueError("Forward not defined in valid function for model: {}".format(self.model_type))
                    

                    loss = self.bce(input=logit, target=y, pos_weight=weights)   # logits bce version, peforms first sigmoid and binary cross entropy on the output
                    losses.append(loss.item())

                        
        # go back to train mode 
        self.model.train()
        
        # return the average loss
        loss_valid = sum(losses)/len(losses)
        print(f"Loss from validation: {loss_valid}")
        return loss_valid

    @duration
    def train(self, additional_name = "", task_type_prog = None, test_loop = False):
        # """ requried the ood data name to recognize the task """
        
        # 1) prepare meta-data
        self._meta_data(task_type_prog= task_type_prog)
        
        # compose train name
        current_date        = date.today().strftime("%d-%m-%Y")   
        train_name          = self.name + "_" + self.model_type + "_"+ additional_name + "_" + current_date
        self.train_name     = train_name
        
        # get paths to save files on model and results
        path_save_model     = self.get_path2SaveModels(train_name  = train_name)   # specify train_name for an additional depth layer in the models file system
        path_save_results   = self.get_path2SaveResults(train_name = train_name)

        print(path_save_model)
        print(path_save_results)
        # 2) prepare the training components
        self.model.train()
        
        # compute the weights for the labels
        self.weights_labels = self.compute_class_weights(verbose=True, positive="ood")
        
        train_dl = DataLoader(self.dataset_train, batch_size= self.batch_size,  num_workers = 8,  shuffle= True,   pin_memory= False)
        valid_dl = DataLoader(self.dataset_valid, batch_size= self.batch_size,  num_workers = 8, shuffle = False,  pin_memory= False) 
        
        # compute number of steps for epoch
        n_steps = len(train_dl)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        # self.scheduler = None
        scaler = GradScaler()
        
        # initialize logger
        logger  = self.init_logger(path_model= path_save_model)
        
        # intialize data structure to keep track of training performance
        loss_epochs     = []
        valid_history   = []
        
        # learned epochs by the model initialization
        self.modelEpochs = 0
        
        # 3) learning loop
        
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            # print(self.classifier.model.large_encoding)
            # define cumulative loss for the current epoch and max/min loss
            loss_epoch = 0; max_loss_epoch = 0; min_loss_epoch = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
            time1 = []
            time2 = []
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dl), total= n_steps):
                
                # if step_idx >= 50: break
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # if step_idx == 20: break
                # zeroing the gradient
                self.optimizer.zero_grad()
                
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(True)
                
                y = y[:,1]                           # take only label for the positive class (fake)
                
                # compute weights for the full batch
                # weights     = T.tensor([self.weights_labels[elem] for elem in y ]).to(self.device)   #TODO check this usage of the class weight, try pos_weight
                
                # compute weight for the positive class
                pos_weight  = T.tensor([self.weights_labels[1]]).to(self.device)

                # int2float and move data to GPU mem                
                y = y.to(self.device).to(T.float32)               # binary int encoding for each sample
               
        
                # print(x.shape)
                # print(y.shape)
                
                # model forward and loss computation
                
                s_1 = time()
                with T.no_grad():  # avoid storage gradient for the classifier
                    out = self._forward_A(x)

                    probs_softmax       = out["probabilities"]
                    encoding            = out["encoding"]
                    residual            = out["residual"]
                    
                if self.use_confidence:
                    confidence = out["confidence"]
                    if self.conf_usage_mode.lower().strip() == "merge":
                        probs_softmax = T.cat((probs_softmax, confidence),dim = 1)
                    elif self.conf_usage_mode.lower().strip() == "alone":
                        probs_softmax = confidence

                time1.append(time()- s_1) 
                
                with autocast():
                    s_2 = time()
                                       
                    probs_softmax.requires_grad_(True)
                    encoding.requires_grad_(True)
                    residual.requires_grad_(True)
                    
                   
                    if self.model_type == "basic" or "encoder" in self.model_type:
                        pass
                        # logit = self._forward_B(prob_softmax, encoding, residual)
                        # logit = T.squeeze(self.model.forward(residual))
                        logit = T.squeeze(self.model.forward(probs_softmax, encoding, residual))
                    else:
                        raise ValueError("Forward not defined in train function for model: {}".format(self.model_type))
                    # print(logit.shape)
                    time2.append(time()- s_2)

                    # print(logit.shape)
                    # print(y.shape)
                
                    loss = self.bce(input=logit, target= y, pos_weight=pos_weight)
                    # print(loss)
    
                 

                # Exit the autocast() context manager
                autocast(enabled=False)
                
                loss_value = loss.item()
                if loss_value>max_loss_epoch    : max_loss_epoch = round(loss_value,4)
                if loss_value<min_loss_epoch    : min_loss_epoch = round(loss_value,4)
                
                # update total loss    
                loss_epoch += loss.item()   # from tensor with single value to int and accumulation
                
                # loss backpropagation
                scaler.scale(loss).backward()
                # loss.backward()
                
                # (Optional) Clip gradients to prevent exploding gradients
                # T.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                
                # compute updates using optimizer
                scaler.step(self.optimizer)
                # self.optimizer.step()

                # update weights through scaler
                scaler.update()
                
                # lr scheduler step 
                self.scheduler.step()
            
            # T.cuda.empty_cache()
            
            print("Time (avg) for forward module A -> ",round(sum(time1)/len(time1),5))   
            print("Time (avg) for forward module B -> ",round(sum(time2)/len(time2),5))
            
            
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # include validation here if needed
            criterion = self.valid(epoch=epoch_idx+1, valid_dl = valid_dl)
            # criterion = 0
            
            valid_history.append(criterion)  

                        
            # create dictionary with info frome epoch: loss + valid, and log it
            epoch_data = {"epoch": last_epoch, "avg_loss": avg_loss, "max_loss": max_loss_epoch, \
                          "min_loss": min_loss_epoch, "valid_loss": criterion}
            logger.log(epoch_data)
            
            # test epochs loop for debug   
            if test_loop and last_epoch == 5: break
        
        
        # log GPU memory statistics during training
        logger.log_mem(T.cuda.memory_summary(device=self.device))
        
        # 4) Save section
        
        # save loss 
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        name_valid_file         = "valid_" + str(last_epoch) + '.png' 
        if test_loop:
            plot_loss(loss_epochs, title_plot = self.name + "_" + self.model_type, path_save = None)
            plot_valid(valid_history, title_plot = self.name + "_" + self.model_type, path_save= None)
        else: 
            plot_loss(loss_epochs, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_results, name_loss_file))
            plot_loss(loss_epochs, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_model  ,name_loss_file), show=False)  # just save, not show
            plot_valid(valid_history, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_results, name_valid_file))
            plot_valid(valid_history, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_model  ,name_valid_file), show=False)  # just save, not show


        # save model
        print("\n\t\t[Saving model]\n")
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_save_model, name_model_file)  # path folder + name file
        saveModel(self.model, path_model_save)
    
        # terminate the logger
        logger.end_log()
        
    def test_risk(self, task_type_prog = None):
        
        # saving folder path
        path_results_folder         = self.get_path2SaveResults(train_name=self.train_name)
        
        # 1) prepare meta-data
        self._meta_data(task_type_prog= task_type_prog)
        self.model.eval()
        
        # 2) prepare test
        test_dl = DataLoader(self.dataset_test, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(test_dl)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # empty lists to store results
        pred_risks = np.empty((0,1), dtype= np.float32)
        dl_labels = np.empty((0,2), dtype= np.int32)   
        
        for idx, (x,y) in tqdm(enumerate(test_dl), total= n_steps):
            
            # to test
            # if idx >= 1: break
            
            x = x.to(self.device)
            with T.no_grad():
                with autocast():
                    
                    out = self._forward_A(x)

                    probs_softmax       = out["probabilities"]
                    encoding            = out["encoding"]
                    residual            = out["residual"]
                    
                    if self.use_confidence:
                        confidence = out["confidence"]
                        if self.conf_usage_mode.lower().strip() == "merge":
                            probs_softmax = T.cat((probs_softmax, confidence),dim = 1)
                        elif self.conf_usage_mode.lower().strip() == "alone":
                            probs_softmax = confidence
                                        
                    # if not basic Abnromality model, do recuction here
                    
                    logit = self._forward_B(probs_softmax, encoding, residual)
                    # risk = T.squeeze(risk)
                    risk =self.sigmoid(logit)
                    
            # to numpy array
            risk   = risk.cpu().numpy()
            y       = y.numpy()
            
            pred_risks = np.append(pred_risks, risk, axis= 0)
            dl_labels = np.append(dl_labels, y, axis= 0)
            
        # divide id risk from  ood risk
        
        id_labels     =  dl_labels[:,0]                # filter by label column
        ood_labels    =  dl_labels[:,1]
        risks_id      = pred_risks[id_labels == 1]         # split forward probabilites between ID adn OOD, still a list of probabilities for each class learned by the model
        risks_ood     = pred_risks[ood_labels == 1]
        
        risks_id = risks_id[:risks_ood.shape[0]]
        # print(risks_id.shape)
        # print(risks_ood.shape)
         
        # compute statistical moments from risks output
        
        conf_all           = np.average(pred_risks)
        
        id_mean_r          =  np.mean(risks_id)
        id_std_r           =  np.std(risks_id)
        # id_entropy         = self.entropy(risks_id)
        # id_mean_e          = np.mean(id_entropy)
        # id_std_e           = np.std(id_entropy) 
    
        ood_mean_r         =  np.mean(risks_ood)
        ood_std_r          =  np.std(risks_ood)
        # ood_entropy        = self.entropy(risks_ood)
        # ood_mean_e         = np.mean(ood_entropy)
        # ood_std_e          = np.std(ood_entropy)

        # in-out of distribution moments
        print("In-Distribution risk                 [mean (confidence ID),std]  -> ", id_mean_r, id_std_r)
        print("Out-Of-Distribution risk             [mean (confidence OOD),std] -> ", ood_mean_r, ood_std_r)
        
        # print("In-Distribution Entropy risk         [mean,std]                  -> ", id_mean_e, id_std_e)
        # print("Out-Of-Distribution Entropy risk     [mean,std]                  -> ", ood_mean_e, ood_std_e)
    
        # normality detection
        print("Normality detection:")
        norm_base_rate = round(100*(risks_id.shape[0]/(risks_id.shape[0] + risks_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(norm_base_rate))
        # print("\tKL divergence (entropy)")
        # kl_norm_aupr, kl_norm_auroc = self.compute_curves(id_entropy, ood_entropy)
        print("\tPrediction probability")
        p_norm_aupr, p_norm_auroc = self.compute_curves(1- risks_id, 1- risks_ood)
        
        # abnormality detection
        print("Abnormality detection:")
        abnorm_base_rate = round(100*(risks_ood.shape[0]/(risks_id.shape[0] + risks_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(abnorm_base_rate))
        # print("\tKL divergence (entropy)")
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(-id_entropy, -ood_entropy, positive_reversed= True)
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-id_entropy, 1-ood_entropy, positive_reversed= True)
        print("\tPrediction probability")
        p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(risks_id, risks_ood, positive_reversed= True)
        # p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-risks_id, 1-risk_ood, positive_reversed= True)
        

        
        # compute fpr95, detection_error and threshold_error 
        metrics_norm = self.compute_metrics_ood(1-risks_id, 1-risks_ood)
        print("OOD metrics:\n", metrics_norm)  
        metrics_abnorm = self.compute_metrics_ood(risks_id, risks_ood, positive_reversed= True, path_save = path_results_folder)
        print("OOD metrics:\n", metrics_abnorm)  
        
        
                # store statistics/metrics in a dictionary
        data = {
            "ID_max_prob": {
                "mean":  float(id_mean_r), 
                "var":   float(id_std_r)
            },
            "OOD_max_prob": {
                "mean": float(ood_mean_r), 
                "var":  float(ood_std_r) 
            },
            # "ID_entropy":   {
            #     "mean": float(id_mean_e), 
            #     "var":  float(id_std_e)
            # },
            # "OOD_entropy":  {
            #     "mean": float(ood_mean_e), 
            #     "var":  float(ood_std_e)
            # },
            
            "normality": {
                "base_rate":        float(norm_base_rate),
                # "KL_AUPR":          float(kl_norm_aupr),
                # "KL_AUROC":         float(kl_norm_auroc),
                "Prob_AUPR":        float(p_norm_aupr),   
                "Prob_AUROC":       float(p_norm_auroc)
            },
            "abnormality":{
                "base_rate":        float(abnorm_base_rate),
                # "KL_AUPR":          float(kl_abnorm_aupr),
                # "KL_AUROC":         float(kl_abnorm_auroc),
                "Prob_AUPR":        float(p_abnorm_aupr),   
                "Prob_AUROC":       float(p_abnorm_auroc),

            },
            "avg_confidence":   float(conf_all),
            "fpr95_normality":            float(metrics_norm['fpr95']),
            "detection_error_normality":  float(metrics_norm['detection_error']),
            "threshold_normality":        float(metrics_norm['thr_de']),
            "fpr95_abnormality":            float(metrics_abnorm['fpr95']),
            "detection_error_abnormality":  float(metrics_abnorm['detection_error']),
            "threshold_abnormality":        float(metrics_abnorm['thr_de'])
            
        }
        
        # save data (JSON)
        if self.name_ood_data is not None:
            name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
            path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
            
            print(path_result_save)
                
            saveJson(path_file = path_result_save, data = data)
        
    def test_threshold(self):
        self.model.eval()
        raise NotImplementedError

class Abnormality_module_ViT(OOD_Classifier):   # model to train necessary 
    """ 
        Modification of Abnormality_module for ViT model, using attention map instead of image reconstruction
    """
    
    def __init__(self, classifier: DFD_BinViTClassifier_v7, scenario:str, model_type: str, useGPU: bool= True, binary_dataset: bool = True,
                 batch_size = 32, use_synthetic:bool = True, extended_ood: bool = False, blind_test: bool = True,
                 balancing_mode: str = "max", ):
        """ 
            ARGS:
            - classifier (DFD_BinViTClassifier_v7): the ViT classifier + Autoencoder (Module A) that produces the input for Module B (abnormality module)
            - scenario (str): choose between: "content", "mix", "group"
            - model_type (str): choose between avaialbe model for the abnrormality module: "basic", "encoder", "encoder_v2", "encoder_v3"
            "encoder_v4"
            - batch_size (str/int): the size of the batch, set defaut to use the assigned from superclass, otherwise the int size. Default is "default".
            - use_synthetic (boolean): choose if use ood data generated from ID data (synthetic) with several techniques, or not. Defaults is True.
            - extended_ood (boolean, optional): This has sense if use_synthetic is set to True. Select if extend the ood data for training, using not only synthetic data. Default is True
            - blind_test (boolean, optional): This has sense if use_synthetic is set to True. Select if use real ood data (True) or synthetized one from In distributiion data. Default is True
            - balancing_mode (string,optinal): This has sense if use_synthethid is set to True and extended_ood is set to True.
            Choose between "max and "all", max mode give a balance number of OOD same as ID, while, all produces more OOD samples than ID. Default is "max"
            
        """
        super(Abnormality_module_ViT, self).__init__(useGPU=useGPU)
        
        # set the classifier (module A)
        self.classifier  = classifier
        self.scenario = scenario
        self.binary_dataset = binary_dataset
        self.model_type = model_type
        
        self.name               = "Abnormality_module"
        self.name_classifier    = self.classifier.classifier_name
        self.train_name         = None
        self._meta_data()
        
        # configuration variables for abnormality module (module B)
        self.augment_data_train = False
        self.loss_name          = "weighted bce"   # binary cross entropy or sigmoid cross entropy (weighted)
        
        # instantiation aux elements
        self.bce     = F.binary_cross_entropy_with_logits   # performs sigmoid internally
        # self.ce      = F.cross_entropy()
        self.sigmoid = F.sigmoid
        self.softmax = F.softmax
        self._build_model()
            
        # training parameters  
        if not batch_size == "dafault":   # default batch size is defined in the superclass 
            self.batch_size             = int(batch_size)
            
        # self.lr                     = 1e-4
        self.lr                     = 1e-3
        self.n_epochs               = 20  # 50
        self.weight_decay           = 1e-3                  # L2 regularization term 
        
        # load data ID/OOD
        if self.binary_dataset:   # binary vs multi-class task
            self.dataset_class = CDDB_binary_Partial
        else:
            self.dataset_class = CDDB_Partial
        
        # Datasets flags
        self.use_synthetic  = use_synthetic    
        self.extended_ood   = extended_ood
        self.blind_test     = blind_test
        if not(balancing_mode in ["max", "all"]):
            raise ValueError('Wrong selection for balancing mode. Choose between "max" or "all".') 
        
        self.balancing_mode = balancing_mode
        
        # Define sets
        if self.use_synthetic:
            self._prepare_data_synt(verbose = True)
        else:
            self._prepare_data(verbose=True)
        
    def _build_model(self):
        # compute shapes for the input
        x_shape = (1, *self.classifier.model.input_shape)
        x = T.rand(x_shape).to(self.device)
        out = self._forward_A(x)
        
        probs_softmax       = out["probabilities"]
        encoding            = out["encoding"]
        residual_flatten    = out["residual"]


        if self.model_type == "encoder_v3_vit":
            self.model = Abnormality_module_Encoder_VIT_v3(probs_softmax.shape, encoding.shape, residual_flatten.shape)

        self.model.to(self.device)
        self.model.eval()
    
    def _meta_data(self, task_type_prog = None):
        """ prepare meta data of the current model: task_type_prog and name_ood_data"""
        
        if task_type_prog is None:
            if self.binary_dataset:
                task_type_prog = 0 # binary-class
            else:
                task_type_prog = 1  # multi-class 
        self.task_type_prog = task_type_prog 
        
        try:
            setting = getScenarioSetting()[self.scenario]
        except:
            ValueError("wrong selection for the scenario, is not possible to retrieve the setting")
        
        # if self.scenario == "content":
        #     self.name_ood_data  = "CDDB_" + self.scenario + "_faces_scenario"
        # else:
        #     self.name_ood_data  = "CDDB_" + self.scenario + "_scenario"
        
        self.name_ood_data  = "CDDB_" + self.scenario + "_" + setting + "_scenario"
        
    def _prepare_data_synt(self,verbose = False):
        """ method used to prepare Dataset class used for both training and testing, synthetizing OOD data for training
        
            ARGS:
            - verbose (boolean, optional): choose to print extra information while loading the data
        """
        
        # synthesis of OOD data (train and valid)
        print("\n\t\t[Loading OOD (synthetized) data]\n")
        ood_data_train_syn    = self.dataset_class(scenario = self.scenario, train = True,  ood = False, augment = False, transform2ood = True)
        tmp        = self.dataset_class(scenario = self.scenario, train = False, ood = False, augment = False, transform2ood = True)
        ood_data_valid_syn , ood_data_test_syn      = sampleValidSet(trainset = ood_data_train_syn, testset= tmp, useOnlyTest = True, verbose = True)
        
        # fetch ID data (train, valid and test)
        print("\n\t\t[Loading ID data]\n")
        id_data_train      = self.dataset_class(scenario = self.scenario, train = True,  ood = False, augment = False, transform2ood = False)
        tmp            = self.dataset_class(scenario = self.scenario, train = False, ood = False, augment = False, transform2ood = False)
        id_data_valid , id_data_test   = sampleValidSet(trainset = id_data_train, testset= tmp, useOnlyTest = True, verbose = True)
        
        
        if verbose:
            print("length ID dataset  (train) -> ",  len(id_data_train))
            print("length ID dataset  (valid) -> ",  len(id_data_valid))
            print("length ID dataset  (test) -> ", len(id_data_test))
            print("length OOD dataset (train) synthetized -> ", len(ood_data_train_syn))
            print("length OOD dataset (valid) synthetized -> ", len(ood_data_valid_syn))

                
        if self.extended_ood:
            print("\n\t\t[Extending OOD data with CDDB samples]\n")
            ood_train_expansion = self.dataset_class(scenario = self.scenario, train = True,   ood = True, augment = False)            
            ood_data_train = mergeDatasets(ood_data_train_syn, ood_train_expansion) 
           
            if verbose: print("length OOD dataset after extension (train) -> ", len(ood_data_train))
            
            # train set: id data train + ood from synthetic ood and expansion)
            self.dataset_train = OOD_dataset(id_data_train, ood_data_train, balancing_mode= self.balancing_mode)
        else:
            # train set: id data train + synthetic ood (id data train transformed in ood)
            self.dataset_train = OOD_dataset(id_data_train, ood_data_train_syn, balancing_mode= self.balancing_mode)
            
        if self.blind_test:
            ood_data_test  = self.dataset_class(scenario = self.scenario, train = False,  ood = True, augment = False)
            if verbose: print("length OOD dataset (test) -> ", len(ood_data_test))
            # test set: id data test + ood data test
            self.dataset_test  = OOD_dataset(id_data_test , ood_data_test,  balancing_mode= self.balancing_mode)
        else:
            self.dataset_test  = OOD_dataset(id_data_test , ood_data_test_syn,  balancing_mode= self.balancing_mode)  # not real ood data but the synthetized one (useful to test the effective learning of the model)
        
        # valid set: id data valid + synthetic ood (id data train transformed in ood)
        self.dataset_valid = OOD_dataset(id_data_valid, ood_data_valid_syn, balancing_mode= self.balancing_mode)
        
        
        if verbose: print("length full dataset (train/valid/test) with balancing -> ", len(self.dataset_train), len(self.dataset_valid), len(self.dataset_test))
        print("\n")
    
    def _prepare_data(self, verbose = False):
        
        """ method used to prepare Dataset class used for both training and testing, both ID and OOD comes from CDDB dataset
        
            ARGS:
           - verbose (boolean, optional): choose to print extra information while loading the data
        """
        
        # fetch ID data (train, valid and test)
        print("\n\t\t[Loading ID data]\n")
        id_data_train      = self.dataset_class(scenario = self.scenario, train = True,  ood = False, augment = False, transform2ood = False)
        tmp            = self.dataset_class(scenario = self.scenario, train = False, ood = False, augment = False, transform2ood = False)
        id_data_valid , id_data_test   = sampleValidSet(trainset = id_data_train, testset= tmp, useOnlyTest = True, verbose = True)
        

        print("\n\t\t[Loading OOD data]\n")
        ood_data_train = self.dataset_class(scenario = self.scenario, train = True,   ood = True, augment = False) 
        tmp  = self.dataset_class(scenario = self.scenario, train = False,  ood = True, augment = False)
        ood_data_valid , ood_data_test   = sampleValidSet(trainset = ood_data_train, testset= tmp, useOnlyTest = True, verbose = True)
        
        
        if verbose:
            print("length ID dataset  (train) -> ",  len(id_data_train))
            print("length ID dataset  (valid) -> ",  len(id_data_valid))
            print("length ID dataset  (test)  -> ", len(id_data_test))
        if verbose: 
            print("length OOD dataset (train) -> ", len(ood_data_train))
            print("length OOD dataset (valid) -> ", len(ood_data_valid))
            print("length OOD dataset (test)  -> ", len(ood_data_test))

                
        # define the OOD detection sets
        self.dataset_train = OOD_dataset(id_data_train, ood_data_train, balancing_mode=self.balancing_mode)
        self.dataset_test  = OOD_dataset(id_data_test , ood_data_valid, balancing_mode=self.balancing_mode)
        self.dataset_valid = OOD_dataset(id_data_valid, ood_data_test , balancing_mode=self.balancing_mode)
        
        if verbose: print("length full dataset (train/valid/test) with balancing -> ", len(self.dataset_train), len(self.dataset_valid), len(self.dataset_test))
        print("\n")
     
    def _hyperParams(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs_max": self.n_epochs,
            "weight_decay": self.weight_decay
                }
    
    def _dataConf(self):
        
        # load not fixed config specs with try-catch

        try:
            input_shape = str(self.model.input_shape)
        except:
            input_shape = "empty"
            
        specified_data =  {
            "date_training": date.today().strftime("%d-%m-%Y"),
            "model": "Abnormality module " + self.model_type,
            "input_shape": input_shape,
            "data_scenario": self.scenario,
            "optimizer": self.optimizer.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__,
            "loss": self.loss_name,
            "grad_scaler": True,                # always true
            "base_augmentation": self.augment_data_train,
            "Use OOD data synthetized":  self.use_synthetic,
            "Use extension OOD from CDDB": self.extended_ood,
            "Balancing mode": self.balancing_mode,

            # dataset lengths 
            "Train Set Samples": len(self.dataset_train),
            "Valid Set Samples": len(self.dataset_valid),
            "Test Set Samples":  len(self.dataset_test),
            
            # dataset distribution
            # "ID train samples": round((1/self.pos_weight_labels[0]) * self.samples_train),
            # "OOD train samples": round((1/self.pos_weight_labels[1]) * self.samples_train)
            "pos_weight_samples": self.pos_weight_labels,
            }
        
        # include auto-inferred data from the model if available
        try:
            concat_dict = lambda x,y: {**x, **y}
            model_data = self.model.getAttributes()
            return concat_dict(specified_data, model_data)
        except:
            return specified_data
        
    def init_logger(self, path_model):
        """
            path_model -> specific path of the current model training
        """
        logger = ExpLogger(path_model=path_model)
        
        # logger for the classifier (module A)
        logger.write_config(self.classifier._dataConf(), name_section="Configuration Classifier")
        logger.write_hyper(self.classifier._hyperParams(), name_section="Hyperparameters Classifier")
        try:
            logger.write_model(self.classifier.model.getSummary(verbose=False), name_section="Model architecture Classifier")
            logger.write_model(self.classifier.autoencoder.getSummary(verbose= False), name_section = "AutoEncoder architecture")
        except:
            print("Impossible to retrieve the model structure for logging")
        
        # logger for the abnormality module  (module B)
        logger.write_config(self._dataConf())
        logger.write_hyper(self._hyperParams())
        try:
            logger.write_model(self.model.getSummary(verbose=False))
        except:
            print("Impossible to retrieve the model structure for logging")
        
        return logger
    
    def _forward_A(self, x, verbose = False):
        """ this method return a dictionary with the all the outputs from the model branches
            keys: "probabilities", "encding", "residual", "confidence"
            the confidence key-value pair is present if and only if is a confidence model (check the name)
        """

        # logits, reconstruction, encoding = self.classifier.model.forward(x)
        
        # with T.no_grad():
        output_model = self.classifier.model.forward(x)  # logits, encoding, att_map
        
        # unpack the output based on the model
        logits          = output_model[0]
        encoding        = output_model[1]
        att_map         = output_model[2]
        
        # generate att_map from autoencoder
        att_map = self.classifier.norm(att_map)
        rec_att_map = self.classifier.autoencoder.forward(att_map)
            
        output = {"encoding": encoding}
        probs_softmax = T.nn.functional.softmax(logits, dim=1)
        
        if verbose: 
            print("prob shape -> ", probs_softmax.shape)
            print("encoding shape -> ",encoding.shape)
        
        # from reconstuction to residual
        residual = T.square(rec_att_map - att_map)
        # residual_flatten = T.flatten(residual, start_dim=1)
        
        output["probabilities"] = probs_softmax
        output["residual"]      = residual
        
        if verbose: 
            print("residual shape ->", residual.shape)
        
        return output
                
    def _forward_B(self, probs_softmax, encoding, residual, verbose = False):
        """ if self.use_confidence is True probs_softmax should include the confidence value (Stacked) """
        
        
        y = self.model.forward(probs_softmax, encoding, residual)
        if verbose: print("y shape", y.shape)
        return y
        
    def forward(self,x):
        
        if self.model is None: raise ValueError("No model has been defined, impossible forwarding the data")
        
        if not(isinstance(x, T.Tensor)):
            x = T.tensor(x)
        
        # handle single image, increasing dimensions simulating a batch
        if len(x.shape) == 3:
            x = x.expand(1,-1,-1,-1)
        elif len(x.shape) <= 2 or len(x.shape) >= 5:
            raise ValueError("The input shape is not compatiple, expected a batch or a single image")
        
        # correct the dtype
        if not (x.dtype is T.float32):
            x = x.to(T.float32)
         
        x = x.to(self.device)
        
        out = self._forward_A(x)

        probs_softmax       = out["probabilities"]
        encoding            = out["encoding"]
        residual            = out["residual"]
                
        logit = self._forward_B(probs_softmax, encoding, residual)
        out   = self.sigmoid(logit)
        
        return  out
    
    def load(self, name_folder, epoch):
        
        print("\n\t\t[Loading model]\n")
        
        self._meta_data()
        
        # save folder of the train (can be used to save new files in models and results)
        self.train_name     = name_folder
        self.modelEpochs    = epoch
        
        # get full path to load
        models_path         = self.get_path2SaveModels()
        path2model          = os.path.join(models_path,  name_folder, str(epoch) + ".ckpt")

        try:
            loadModel(self.model, path2model)
            self.model.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except Exception as e:
            print(e)
            print("No model: {} found for the epoch: {} in the folder: {}".format(name_folder, epoch, path2model))
    
    def valid(self, epoch, valid_dl):
        """
            validation method used mainly for the Early stopping training
        """
        print (f"Validation for the epoch: {epoch} ...")
        
        # set temporary evaluation mode and empty cuda cache
        self.model.eval()
        T.cuda.empty_cache()
        
        # list of losses
        losses = []

        
        for (x,y) in tqdm(valid_dl):
            
            x = x.to(self.device)
            # y = y.to(self.device).to(T.float32)
            
            # take only label for the positive class (fake)
            y = y[:,1]
            
            # compute monodimensional weights for the full batch
            # weights = T.tensor([self.pos_weight_labels[elem] for elem in y ]).to(self.device)
            pos_weight  = T.tensor(self.pos_weight_labels).to(self.device)
            
            y = y.to(self.device).to(T.float32)

            
            with T.no_grad():
                # with autocast():
                out = self._forward_A(x)

                probs_softmax       = out["probabilities"]
                encoding            = out["encoding"]
                residual            = out["residual"]
                

                with autocast():
                    if self.model_type == "basic" or "encoder" in self.model_type:
                        # logit = self._forward_B(prob_softmax, encoding, residual)
                        # logit = T.squeeze(logit)
                        logit = T.squeeze(self.model.forward(probs_softmax, encoding, residual))
                    else:
                        raise ValueError("Forward not defined in valid function for model: {}".format(self.model_type))
                    

                    loss = self.bce(input=logit, target=y, pos_weight=pos_weight)   # logits bce version, peforms first sigmoid and binary cross entropy on the output
                losses.append(loss.item())

                        
        # go back to train mode 
        self.model.train()
        
        # return the average loss
        loss_valid = sum(losses)/len(losses)
        print(f"Loss from validation: {loss_valid}")
        return loss_valid

    @duration
    def train(self, additional_name = "", task_type_prog = None, test_loop = False):
        # """ requried the ood data name to recognize the task """
        
        # 1) prepare meta-data
        self._meta_data(task_type_prog= task_type_prog)
        
        # compose train name
        current_date        = date.today().strftime("%d-%m-%Y")   
        train_name          = self.name + "_" + self.model_type + "_"+ additional_name + "_" + current_date
        self.train_name     = train_name
        
        # get paths to save files on model and results
        path_save_model     = self.get_path2SaveModels(train_name  = train_name)   # specify train_name for an additional depth layer in the models file system
        path_save_results   = self.get_path2SaveResults(train_name = train_name)

        print(path_save_model)
        print(path_save_results)
        # 2) prepare the training components
        self.model.train()
        
        # compute the weights for the labels
        self.pos_weight_labels = self.compute_class_weights(verbose=True, positive="ood", only_positive_weight= True)
        
        train_dl = DataLoader(self.dataset_train, batch_size= self.batch_size,  num_workers = 8,  shuffle= True,   pin_memory= False)
        valid_dl = DataLoader(self.dataset_valid, batch_size= self.batch_size,  num_workers = 8, shuffle = False,  pin_memory= False) 
        
        # compute number of steps for epoch
        n_steps = len(train_dl)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        # self.scheduler = None
        scaler = GradScaler()
        
        # initialize logger
        logger  = self.init_logger(path_model= path_save_model)
        
        # intialize data structure to keep track of training performance
        loss_epochs     = []
        valid_history   = []
        
        # learned epochs by the model initialization
        self.modelEpochs = 0
        
        # 3) learning loop
        
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            # print(self.classifier.model.large_encoding)
            # define cumulative loss for the current epoch and max/min loss
            loss_epoch = 0; max_loss_epoch = 0; min_loss_epoch = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
            time1 = []
            time2 = []
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dl), total= n_steps):
                
                # if step_idx >= 50: break
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # if step_idx == 20: break
                # zeroing the gradient
                self.optimizer.zero_grad()
                
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                # x.requires_grad_(True)
                y = y[:,1]                           # take only label for the positive class (fake)
                
                # compute weights for the full batch
                # weights     = T.tensor([self.weights_labels[elem] for elem in y ]).to(self.device)   #TODO check this usage of the class weight, try pos_weight
                
                # compute weight for the positive class
                # pos_weight  = T.tensor([self.weights_labels[1]]).to(self.device)
                pos_weight  = T.tensor(self.pos_weight_labels).to(self.device)
                # print(pos_weight.shape)

                # int2float and move data to GPU mem                
                y = y.to(self.device).to(T.float32)               # binary int encoding for each sample
            
                # model forward and loss computation
                
                s_1 = time()
                with T.no_grad():  # avoid storage gradient for the classifier
                    out = self._forward_A(x)

                    probs_softmax       = out["probabilities"]
                    encoding            = out["encoding"]
                    residual            = out["residual"]
                    
                time1.append(time()- s_1) 
                
                # with autocast():
                s_2 = time()
                                    
                probs_softmax.requires_grad_(True)
                encoding.requires_grad_(True)
                residual.requires_grad_(True)
                
                with autocast():
                    if self.model_type == "basic" or "encoder" in self.model_type:

                        logit = T.squeeze(self.model.forward(probs_softmax, encoding, residual))
                        
                    else:
                        raise ValueError("Forward not defined in train function for model: {}".format(self.model_type))
                    # print(logit.shape)
                    time2.append(time()- s_2)


                    loss = self.bce(input=logit, target= y, pos_weight=pos_weight)
                # print(loss)

                

                # Exit the autocast() context manager
                # autocast(enabled=False)
                
                loss_value = loss.item()
                if loss_value>max_loss_epoch    : max_loss_epoch = round(loss_value,4)
                if loss_value<min_loss_epoch    : min_loss_epoch = round(loss_value,4)
                
                # update total loss    
                loss_epoch += loss.item()   # from tensor with single value to int and accumulation
                
                # loss backpropagation
                scaler.scale(loss).backward()
                # loss.backward()
                
                # (Optional) Clip gradients to prevent exploding gradients
                # T.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                
                # compute updates using optimizer
                scaler.step(self.optimizer)
                # self.optimizer.step()

                # update weights through scaler
                scaler.update()
                
                # lr scheduler step 
                self.scheduler.step()
            
            # T.cuda.empty_cache()
            
            print("Time (avg) for forward module A -> ",round(sum(time1)/len(time1),5))   
            print("Time (avg) for forward module B -> ",round(sum(time2)/len(time2),5))
            
            
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # include validation here if needed
            criterion = self.valid(epoch=epoch_idx+1, valid_dl = valid_dl)
            # criterion = 0
            
            valid_history.append(criterion)  

                        
            # create dictionary with info frome epoch: loss + valid, and log it
            epoch_data = {"epoch": last_epoch, "avg_loss": avg_loss, "max_loss": max_loss_epoch, \
                          "min_loss": min_loss_epoch, "valid_loss": criterion}
            logger.log(epoch_data)
            
            # test epochs loop for debug   
            if test_loop and last_epoch == 5: break
        
        
        # log GPU memory statistics during training
        logger.log_mem(T.cuda.memory_summary(device=self.device))
        
        # 4) Save section
        
        # save loss 
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        name_valid_file         = "valid_" + str(last_epoch) + '.png' 
        if test_loop:
            plot_loss(loss_epochs, title_plot = self.name + "_" + self.model_type, path_save = None)
            plot_valid(valid_history, title_plot = self.name + "_" + self.model_type, path_save= None)
        else: 
            plot_loss(loss_epochs, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_results, name_loss_file))
            plot_loss(loss_epochs, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_model  ,name_loss_file), show=False)  # just save, not show
            plot_valid(valid_history, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_results, name_valid_file))
            plot_valid(valid_history, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_model  ,name_valid_file), show=False)  # just save, not show


        # save model
        print("\n\t\t[Saving model]\n")
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_save_model, name_model_file)  # path folder + name file
        saveModel(self.model, path_model_save)
    
        # terminate the logger
        logger.end_log()
        
    def test_risk(self, task_type_prog = None):
        
        # saving folder path
        path_results_folder         = self.get_path2SaveResults(train_name=self.train_name)
        
        # 1) prepare meta-data
        self._meta_data(task_type_prog= task_type_prog)
        self.model.eval()
        
        # 2) prepare test
        test_dl = DataLoader(self.dataset_test, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(test_dl)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # empty lists to store results
        pred_risks = np.empty((0,1), dtype= np.float32)
        dl_labels = np.empty((0,2), dtype= np.int32)   
        
        for idx, (x,y) in tqdm(enumerate(test_dl), total= n_steps):
            
            # to test
            # if idx >= 1: break
            
            x = x.to(self.device)
            with T.no_grad():
                # with autocast():
                    
                out = self._forward_A(x)

                probs_softmax       = out["probabilities"]
                encoding            = out["encoding"]
                residual            = out["residual"]
                                    
                # if not basic Abnromality model, do recuction here
                logit = self._forward_B(probs_softmax, encoding, residual)
                # risk = T.squeeze(risk)
                risk =self.sigmoid(logit)
                    
            # to numpy array
            risk   = risk.cpu().numpy()
            y       = y.numpy()
            
            pred_risks = np.append(pred_risks, risk, axis= 0)
            dl_labels = np.append(dl_labels, y, axis= 0)
            
        # divide id risk from  ood risk
        
        id_labels     =  dl_labels[:,0]                # filter by label column
        ood_labels    =  dl_labels[:,1]
        risks_id      = pred_risks[id_labels == 1]         # split forward probabilites between ID adn OOD, still a list of probabilities for each class learned by the model
        risks_ood     = pred_risks[ood_labels == 1]
        
        risks_id = risks_id[:risks_ood.shape[0]]
        # print(risks_id.shape)
        # print(risks_ood.shape)
         
        # compute statistical moments from risks output
        
        conf_all           = np.average(pred_risks)
        
        id_mean_r          =  np.mean(risks_id)
        id_std_r           =  np.std(risks_id)
        # id_entropy         = self.entropy(risks_id)
        # id_mean_e          = np.mean(id_entropy)
        # id_std_e           = np.std(id_entropy) 
    
        ood_mean_r         =  np.mean(risks_ood)
        ood_std_r          =  np.std(risks_ood)
        # ood_entropy        = self.entropy(risks_ood)
        # ood_mean_e         = np.mean(ood_entropy)
        # ood_std_e          = np.std(ood_entropy)

        # in-out of distribution moments
        print("In-Distribution risk                 [mean (confidence ID),std]  -> ", id_mean_r, id_std_r)
        print("Out-Of-Distribution risk             [mean (confidence OOD),std] -> ", ood_mean_r, ood_std_r)
        
        # print("In-Distribution Entropy risk         [mean,std]                  -> ", id_mean_e, id_std_e)
        # print("Out-Of-Distribution Entropy risk     [mean,std]                  -> ", ood_mean_e, ood_std_e)
    
        # normality detection
        print("Normality detection:")
        norm_base_rate = round(100*(risks_id.shape[0]/(risks_id.shape[0] + risks_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(norm_base_rate))
        # print("\tKL divergence (entropy)")
        # kl_norm_aupr, kl_norm_auroc = self.compute_curves(id_entropy, ood_entropy)
        print("\tPrediction probability")
        p_norm_aupr, p_norm_auroc = self.compute_curves(1- risks_id, 1- risks_ood)
        
        # abnormality detection
        print("Abnormality detection:")
        abnorm_base_rate = round(100*(risks_ood.shape[0]/(risks_id.shape[0] + risks_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(abnorm_base_rate))
        # print("\tKL divergence (entropy)")
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(-id_entropy, -ood_entropy, positive_reversed= True)
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-id_entropy, 1-ood_entropy, positive_reversed= True)
        print("\tPrediction probability")
        p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(risks_id, risks_ood, positive_reversed= True)
        # p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-risks_id, 1-risk_ood, positive_reversed= True)
        

        
        # compute fpr95, detection_error and threshold_error 
        metrics_norm = self.compute_metrics_ood(1-risks_id, 1-risks_ood)
        print("OOD metrics:\n", metrics_norm)  
        metrics_abnorm = self.compute_metrics_ood(risks_id, risks_ood, positive_reversed= True, path_save = path_results_folder)
        print("OOD metrics:\n", metrics_abnorm)  
        
        
                # store statistics/metrics in a dictionary
        data = {
            "ID_max_prob": {
                "mean":  float(id_mean_r), 
                "var":   float(id_std_r)
            },
            "OOD_max_prob": {
                "mean": float(ood_mean_r), 
                "var":  float(ood_std_r) 
            },
            # "ID_entropy":   {
            #     "mean": float(id_mean_e), 
            #     "var":  float(id_std_e)
            # },
            # "OOD_entropy":  {
            #     "mean": float(ood_mean_e), 
            #     "var":  float(ood_std_e)
            # },
            
            "normality": {
                "base_rate":        float(norm_base_rate),
                # "KL_AUPR":          float(kl_norm_aupr),
                # "KL_AUROC":         float(kl_norm_auroc),
                "Prob_AUPR":        float(p_norm_aupr),   
                "Prob_AUROC":       float(p_norm_auroc)
            },
            "abnormality":{
                "base_rate":        float(abnorm_base_rate),
                # "KL_AUPR":          float(kl_abnorm_aupr),
                # "KL_AUROC":         float(kl_abnorm_auroc),
                "Prob_AUPR":        float(p_abnorm_aupr),   
                "Prob_AUROC":       float(p_abnorm_auroc),

            },
            "avg_confidence":   float(conf_all),
            "fpr95_normality":            float(metrics_norm['fpr95']),
            "detection_error_normality":  float(metrics_norm['detection_error']),
            "threshold_normality":        float(metrics_norm['thr_de']),
            "fpr95_abnormality":            float(metrics_abnorm['fpr95']),
            "detection_error_abnormality":  float(metrics_abnorm['detection_error']),
            "threshold_abnormality":        float(metrics_abnorm['thr_de'])
            
        }
        
        # save data (JSON)
        if self.name_ood_data is not None:
            name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
            path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
            
            print(path_result_save)
                
            saveJson(path_file = path_result_save, data = data)
        
    def test_threshold(self):
        self.model.eval()
        raise NotImplementedError
    
    
if __name__ == "__main__":
    #                           [Start test section] 
    
    # [1] load deep fake classifier
    # choose classifier model as module A associated with a certain scenario
    classifier_model = 2
    scenario = "content"

    if classifier_model == 0:      # Unet + classifiier 
        
        classifier_name = "faces_Unet4_Scorer112p_v4_03-12-2023"
        classifier_type = "Unet4_Scorer"
        classifier_epoch = 73
        classifier = DFD_BinClassifier_v4(scenario="content", model_type=classifier_type)
        classifier.load(classifier_name, classifier_epoch)
        resolution = "112p"
    
    elif classifier_model == 1:   # Unet + classifiier + confidence
    
        classifier_name = "faces_Unet4_Scorer_Confidence_112p_v5_02-01-2024"
        classifier_type = "Unet4_Scorer_Confidence"
        classifier_epoch = 98
        classifier = DFD_BinClassifier_v5(scenario="content", model_type=classifier_type)
        classifier.load(classifier_name, classifier_epoch)
        conf_usage_mode = "ignore" # ignore, merge or alone
        resolution = "112p"
    
    elif classifier_model == 2:         # ViT + Autoencoder
        # classifier_name = "faces_ViTEA_timm_v7_07-02-2024"
        # classifier_type = "ViTEA_timm"
        # classifier_epoch = 21
        # scenario = "content"
        # classifier = DFD_BinViTClassifier_v7(scenario="content", model_type=classifier_type)
        # classifier.load(classifier_name, classifier_epoch)
        # resolution = "224p"
        classifier_name     = "faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_13-02-2024"
        classifier_type     = "ViTEA_timm"
        autoencoder_type    = "vae"
        prog_model_timm     = 3 # (tiny DeiT)
        classifier_epoch    = 25
        autoencoder_epoch   = 25
        classifier = DFD_BinViTClassifier_v7(scenario="content", model_type=classifier_type, autoencoder_type = autoencoder_type,\
                                             prog_pretrained_model= prog_model_timm)
        # load classifier & autoencoder
        classifier.load_both(classifier_name, classifier_epoch, autoencoder_epoch)
        resolution = "224p"
    
    
    
    
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
      
    def test_baseline_content_faces():
        name_ood_data_content  = "CDDB_content_faces_scenario"
        
        # laod id data test
        id_data_train      = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment = False)
        id_data_test       = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment = False)
        _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
        
        # load ood data test
        ood_data_test  = CDDB_binary_Partial(scenario = "content", train = False,  ood = True, augment = False)
        
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
    
    def test_baselineOdin_content_faces():
        name_ood_data_content  = "CDDB_content_faces_scenario"

        # laod id data test
        id_data_train      = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment = False)
        id_data_test       = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment = False)
        _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
        
        # load ood data test
        ood_data_test  = CDDB_binary_Partial(scenario = "content", train = False,  ood = True, augment = False)
        
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
    
    def train_abn_encoder(type_encoder = "encoder"):
        
        if classifier_model == 2:
            abn = Abnormality_module_ViT(classifier, scenario = scenario, model_type= "encoder_v3_vit")
        else: 
            abn = Abnormality_module(classifier, scenario = scenario, model_type= type_encoder, conf_usage_mode = conf_usage_mode)
        # abn.train(additional_name= resolution + "_ignored_confidence" , test_loop=False)
        abn.train(additional_name= resolution, test_loop=False)
        
    def train_extended_abn_encoder(type_encoder = "encoder"):
        """ uses extended OOD data"""
        if classifier_model == 2:
            abn = Abnormality_module_ViT(classifier, scenario = scenario, model_type= "encoder_v3_vit", extended_ood = True)
        else: 
            abn = Abnormality_module(classifier, scenario = scenario, model_type= type_encoder, extended_ood = True,  conf_usage_mode = conf_usage_mode)
        abn.train(additional_name= resolution + "_extendedOOD", test_loop=False)
        
    def train_nosynt_abn_encoder(type_encoder = "encoder"):
        """ uses extended OOD data"""
        
        if classifier_model == 2:
            abn = Abnormality_module_ViT(classifier, scenario = scenario, model_type="encoder_v3_vit", use_synthetic= False)
        else: 
            abn = Abnormality_module(classifier, scenario = scenario, model_type=type_encoder, use_synthetic= False,  conf_usage_mode = conf_usage_mode)
        abn.train(additional_name= resolution + "_nosynt", test_loop=False)
    
    def train_full_extended_abn_encoder(type_encoder = "encoder"):
        
        """ uses extended OOD data"""
        if classifier_model == 2:
            abn = Abnormality_module_ViT(classifier, scenario = scenario, model_type= "encoder_v3_vit", extended_ood = True, balancing_mode="all")
        else: 
            abn = Abnormality_module(classifier, scenario = scenario, model_type= type_encoder, extended_ood = True, balancing_mode="all",  conf_usage_mode = conf_usage_mode)
        abn.train(additional_name = resolution + "_fullExtendedOOD", test_loop=False)
    
    def test_abn(name_model, epoch, type_encoder):
        
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
            abn = Abnormality_module_ViT(classifier, scenario=scenario, model_type="encoder_v3_vit")
        else: 
            abn = Abnormality_module(classifier, scenario=scenario, model_type=type_encoder,  conf_usage_mode = conf_usage_mode)
            
        abn.load(name_model, epoch)
        
        # test_forward()
    
        # launch test with non-thr metrics
        abn.test_risk()
    
    # train_abn_encoder()
    # test_abn("Abnormality_module_encoder_v3_vit_224p_08-02-2024", 20, None)
    
    
    
    
    pass
    #                           [End test section] 
   
    """ 
            Past test/train launched: 
            
    classifier: faces_Unet4_Scorer112p_v4_03-12-2023
    
                                    BASELINE    
        test_baseline_implementation()
        test_baseline_facesCDDB_CIFAR()
        test_baseline_content_faces()
        
                                    BASELINE + ODIN
        test_baselineOdin_facesCDDB_CIFAR()
        test_baselineOdin_content_faces()
                                    
                                    ABNORMALITY MODULE BASIC  (Synthetic ood data, no extension)
        train_abn_basic()
        
        test_abn_content_faces("Abnormality_module_basic_112p_05-12-2023", 30,"basic")
        
                                    ABNORMALITY MODULE ENCODER  (Synthetic ood data, no extension)
        train_abn_encoder(type_encoder = "encoder" )
        train_abn_encoder(type_encoder = "encoder_v2" )
        train_abn_encoder(type_encoder = "encoder_v3" )
        train_abn_encoder(type_encoder = "encoder_v4" )
        
        test_abn_content_faces("Abnormality_module_encoder_112p_19-12-2023", 30,"encoder")
        test_abn_content_faces("Abnormality_module_encoder_v2_112p_19-12-2023", 50,"encoder_v2")
        test_abn_content_faces("Abnormality_module_encoder_v3_112p_19-12-2023", 50,"encoder_v3")
        test_abn_content_faces("Abnormality_module_encoder_v4_112p_19-12-2023", 50,"encoder_v4")
        
                                    ABNORMALITY MODULE ENCODER  (Synthetic ood data, + extension, max merging)
        train_extended_abn_encoder(type_encoder = encoder_v4)
        train_extended_abn_encoder(type_encoder= "encoder")
        train_extended_abn_encoder(type_encoder= "encoder_v2")
        train_extended_abn_encoder(type_encoder= "encoder_v3")         
        
        test_abn_content_faces("Abnormality_module_encoder_112p_extendedOOD_26-12-2023", 50,"encoder")    
        test_abn_content_faces("Abnormality_module_encoder_v2_112p_extendedOOD_26-12-2023", 50,"encoder_v2")    
        test_abn_content_faces("Abnormality_module_encoder_v3_112p_extendedOOD_26-12-2023", 50,"encoder_v3")      
        test_abn_content_faces("Abnormality_module_encoder_v4_112p_extendedOOD_19-12-2023", 50,"encoder_v4")
        
                                    ABNORMALITY MODULE ENCODER  (CDDB OOD data)
        train_nosynt_abn_encoder(type_encoder= "encoder")
        train_nosynt_abn_encoder(type_encoder= "encoder_v2")
        train_nosynt_abn_encoder(type_encoder= "encoder_v3")                 
        train_nosynt_abn_encoder(type_encoder= "encoder_v4")
        
        test_abn_content_faces("Abnormality_module_encoder_112p_nosynt_25-12-2023", 50,"encoder")    
        test_abn_content_faces("Abnormality_module_encoder_v2_112p_nosynt_25-12-2023", 50,"encoder_v2")    
        test_abn_content_faces("Abnormality_module_encoder_v3_112p_nosynt_25-12-2023", 50,"encoder_v3")      
        test_abn_content_faces("Abnormality_module_encoder_v4_112p_nosynt_25-12-2023", 50,"encoder_v4")   

                                    ABNORMALITY MODULE ENCODER  (Synthetic ood data, + extension, all merging)
        train_full_extended_abn_encoder("encoder")
        train_full_extended_abn_encoder("encoder_v2")
        train_full_extended_abn_encoder("encoder_v3")
        train_full_extended_abn_encoder("encoder_v4")
    
        test_abn_content_faces("Abnormality_module_encoder_112p_fullExtendedOOD_27-12-2023",50, "encoder")
        test_abn_content_faces("Abnormality_module_encoder_v2_112p_fullExtendedOOD_28-12-2023",50, "encoder_v2")
        test_abn_content_faces("Abnormality_module_encoder_v3_112p_fullExtendedOOD_28-12-2023",50, "encoder_v3")
        test_abn_content_faces("Abnormality_module_encoder_v4_112p_fullExtendedOOD_29-12-2023",50, "encoder_v4")
    
    classifier: faces_Unet4_Scorer_Confidence_112p_v5_02-01-2024:
    
                                    CONFIDENCE DETECTOR
        test_confidenceDetector_facesCDDB_CIFAR()
        
        
                                    ABNORMALITY MODULE ENCODER  (Synthetic ood data, no extension)
        train_abn_encoder(type_encoder= "encoder")
        train_abn_encoder(type_encoder= "encoder_v3")
        train_abn_encoder(type_encoder="encoder_v3")   # conf_usage_mode = "alone"
        train_abn_encoder(type_encoder= "encoder_v3")  
        
        test_abn("Abnormality_module_encoder_112p_09-01-2024", 50, "encoder")
        test_abn("Abnormality_module_encoder_v3_112p_09-01-2024", 50, "encoder_v3")
        test_abn("Abnormality_module_encoder_v3_112p_11-01-2024",50,"encoder_v3")                        # conf_usage_mode = "alone"
        test_abn("Abnormality_module_encoder_v3_112p_ignored_confidence_11-01-2024",50,"encoder_v3")     # conf_usage_mode = "ignore"
                                    
                                    ABNORMALITY MODULE ENCODER  (Synthetic ood data, + extension, max merging)
        train_extended_abn_encoder(type_encoder= "encoder_v3")
        
        test_abn("Abnormality_module_encoder_v3_112p_extendedOOD_09-01-2024", 50, "encoder_v3")
                                    
                                    ABNORMALITY MODULE ENCODER  (CDDB OOD data)
        train_nosynt_abn_encoder(type_encoder= "encoder_v3")
        
        test_abn("Abnormality_module_encoder_v3_112p_nosynt_09-01-2024", 50, "encoder_v3")
                                
                                    ABNORMALITY MODULE ENCODER  (Synthetic ood data, + extension, all merging)
        train_full_extended_abn_encoder(type_encoder= "encoder_v3")
        
        test_abn("Abnormality_module_encoder_v3_112p_fullExtendedOOD_09-01-2024", 50, "encoder_v3")
    
    
    classifier: "faces_ViTEA_timm_DeiT_tiny_separateTrain_v7_13-02-2024":
    
                                    ABNORMALITY MODULE ENCODER  (Synthetic ood data, no extension)

                                    
                                    ABNORMALITY MODULE ENCODER  (Synthetic ood data, + extension, max merging)

                                    
                                    ABNORMALITY MODULE ENCODER  (CDDB OOD data)

                                
                                    ABNORMALITY MODULE ENCODER  (Synthetic ood data, + extension, all merging)

    
    
    
    
    """
