import  os
import  torch               as T
import  numpy               as np
import  math
from    torch.nn            import functional as F
from    torch.utils.data    import DataLoader
from    torch.cuda.amp      import autocast
from    tqdm                import tqdm
from    datetime            import date
from    sklearn.metrics     import precision_recall_curve, auc, roc_auc_score
from    torch.optim         import Adam, lr_scheduler
from    torch.cuda.amp      import GradScaler, autocast
# local import
from    dataset             import CDDB_binary, CDDB_binary_Partial, CDDB, CDDB_Partial, OOD_dataset, getCIFAR100_dataset, getMNIST_dataset, getFMNIST_dataset
from    experiments         import MNISTClassifier_keras
from    bin_classifier      import DFD_BinClassifier_v1, DFD_BinClassifier_v4
from    models              import Abnormality_module_Basic
from    utilities           import saveJson, loadJson, metrics_binClass, metrics_OOD, print_dict, showImage, check_folder, sampleValidSet, \
                            mergeDatasets, ExpLogger, loadModel, saveModel, duration, plot_loss

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
        
        self.batch_size   = 32
        
    #                                       math/statistics aux functions
    
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
        
    def odin_perturbations(self,x, classifier, is_batch = True, loss_function = F.cross_entropy, epsilon = 12e-4, t = 1000):
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
            
        x.requires_grad_(True)  # Enable gradient computation for the input
        
        # forward and loss
        _, _, logits = classifier.forward(x)
        
        if False: 
            logits = logits/t
            
            if is_batch:
                loss = loss_function(logits, T.argmax(logits, dim=-1))
            else:
                loss = loss_function(logits, T.argmax(logits))  # Assuming cross-entropy loss for illustration
            
            # backprogation
            # classifier.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()         # Compute gradients

            # gradient based perturbation
            perturbation = epsilon * T.sign(x.grad.data)    # This ensures that the perturbation is added in the direction that increases the loss, 
            perturbed_x = x + perturbation              # making the model more sensitive to variations in the input data during inference.

        # classifier.optimizer.zero_grad()  # Clear previous gradients
        
        # Apply temperature scaling to logits
        logits /= t
        
        # Calculate softmax probabilities
        # probabilities = F.softmax(logits, dim=1)
        
        # Calculate the derivative of the cross-entropy loss with respect to the input
        gradients = T.autograd.grad(outputs=logits, inputs=x,
                                        grad_outputs=T.ones_like(logits),
                                        retain_graph=False, create_graph=False)[0]
        
        # Calculate the perturbed input
        perturbed_x = x - epsilon * gradients.sign()
        
        
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

    def compute_metrics_ood(self, id_data, ood_data, positive_reversed = False):
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
        metrics_ood = metrics_OOD(targets=target, pred_probs= predictions)
        return metrics_ood 
    
    # path utilities
    
    def get_path2SaveResults(self, train_name = None):
        """ Return the path to the folder to save results both for OOD metrics and ID/OOD bin classification"""
        
        name_task                   = self.types_classifier[self.task_type_prog]
        path_results_task           = os.path.join(self.path_results, name_task)
        path_results_method         = os.path.join(path_results_task, self.name)
        path_results_ood_data       = os.path.join(path_results_method, "ood_" + self.name_ood_data)
        path_results_classifier     = os.path.join(path_results_ood_data, self.name_classifier) 
        
        if train_name is not None:   
            path_results_folder         = os.path.join(path_results_classifier,train_name)    
        
        # prepare file-system
        check_folder(path_results_task)
        check_folder(path_results_method)
        check_folder(path_results_ood_data)
        check_folder(path_results_classifier)
        if train_name is not None:
            check_folder(path_results_folder)
        
        if train_name is not None:
            return path_results_folder
        else: 
            return path_results_classifier
    
    def get_path2SaveModels(self, train_name = None):
        name_task                   = self.types_classifier[self.task_type_prog]
        path_models_task            = os.path.join(self.path_models, name_task)
        path_models_method          = os.path.join(path_models_task, self.name)
        path_models_ood_data        = os.path.join(path_models_method, "ood_"+self.name_ood_data)
        path_models_classifier      = os.path.join(path_models_ood_data, self.name_classifier)    
        if train_name is not None:   
            path_models_folder          = os.path.join(path_models_classifier, train_name)
        
        # prepare file-system
        check_folder(path_models_task)
        check_folder(path_models_method)
        check_folder(path_models_ood_data)
        check_folder(path_models_classifier)
        if train_name is not None:
            check_folder(path_models_folder)
        
        if train_name is not None:
            return path_models_folder
        else:
            return path_models_classifier
    
class Baseline(OOD_Classifier):         # No model training necessary (Empty model forlder)
    """
        OOD detection baseline using softmax probability
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
        """ testing function using probabilty-based metrics, computing OOD metrics that are not threshold related


        """
        
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
            "fpr95":            float(metrics_norm['fpr95']),
            "detection_error":  float(metrics_norm['detection_error']),
            "threshold":        float(metrics_norm['thr_de'])
            
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
        
    def test_threshold(self, thr_type = "fpr95"):
        """
            This function compute metrics (binary classification ID/OOD) that are threshold related (discriminator)
            
                Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset used as ood data, is is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            thr_type (str): choose which kind of threhsold use between "fpr95" (fpr at tpr 95%) or "avg_confidence", or "thr_de",  Default is "fpr95".
            
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
        if thr_type == "fpr95":
            threshold = data['fpr95']
        elif thr_type == "thr_de":
            threshold = data['threshold']
        else:                                       # use avg max prob confidence as thr (proven to be misleading)
            threshold = data['avg_confidence']
        
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
            if prob < threshold: pred.append(1)  # OOD
            else: pred.append(0)                 # ID
        
        # get the list with the binary labels
        pred = np.array(pred)
        target = test_labels[:,1]
        
        # compute and save metrics 
        name_resultClass_file  = 'metrics_ood_classification_{}.json'.format(self.name_ood_data)
        metrics_class =  metrics_binClass(preds = pred, targets= target, pred_probs = None, path_save = path_results_folder, name_ood_file = name_resultClass_file)
        
        print(metrics_class)

    def forward(self, x, thr_type = "fpr95"):
        """ discriminator forward

        Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset/CDDB scenario used as ood data, if is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            thr_type (str): choose which kind of threhsold use between "fpr95" (fpr at tpr 95%) or "avg_confidence", or "thr_de",  Default is "fpr95".
            
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
            
        if thr_type == "fpr95":
            threshold = data['fpr95']
        elif thr_type == "thr_de":
            threshold = data['threshold']
        else:                                      
            threshold = data['avg_confidence']
        
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
        pred = np.where(condition= maximum_prob < threshold, x=1, y=0)  # if true set x otherwise set y
        return pred

# TODO check odin correctness
class Baseline_ODIN(OOD_Classifier):        # No model training necessary (Empty model forlder)
    """
        OOD detection baseline using softmax probability + ODIN framework
    """
    def __init__(self, classifier, name_ood_data,  task_type_prog, id_data_test , ood_data_test , id_data_train = None, ood_data_train = None, useGPU = True):
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
        # set the classifier
        self.classifier  = classifier
        self.task_type_prog = task_type_prog
        self.name_ood_data  = name_ood_data
        # load the Pytorch dataset here
        try:
            self.dataset_test  = OOD_dataset(self.id_data_test, self.ood_data_test, balancing_mode = "max")
        except:
            print("Dataset data is not valid, please use instances of class torch.Dataset")
        
        # name of the classifier
        self.name = "ODIN+baseline"
        self.name_classifier = self.classifier.classifier_name
        
        
    def test_probabilties(self):
        """ testing function using probabilty-based metrics, computing OOD metrics that are not threshold related
        """
        
        # define the dataloader 
        id_ood_dataloader   = DataLoader(self.dataset_test,  batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # use model on selected device
        self.classifier.model.to(self.device)
        
        # define empty list to store outcomes
        pred_logits = np.empty((0,2), dtype= np.float32)
        dl_labels = np.empty((0,2), dtype= np.int32)            # dataloader labels, binary one-hot encoding, ID -> [1,0], OOD -> [0,1]
        
        for idx, (x,y) in tqdm(enumerate(id_ood_dataloader), total= len(id_ood_dataloader)):
            
            # to test
            # if idx >= 5: break
            
            x = x.to(self.device)
            x = self.odin_perturbations(x,self.classifier, is_batch=True)
            
            with T.no_grad():
                with autocast():
                    _ ,_, logits =self.classifier.forward(x)
                    
            # to numpy array
            logits  = logits.cpu().numpy()
            y       = y.numpy()
                
            pred_logits = np.append(pred_logits, logits, axis= 0)
            dl_labels = np.append(dl_labels, y, axis= 0)
            
        # to softmax probabilities
        probs = self.softmax_temperature(pred_logits)
   
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
        kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(-entropy_id, -entropy_ood, positive_reversed= True)
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-entropy_id, 1-entropy_ood, positive_reversed= True)
        print("\tPrediction probability")
        p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(-maximum_prob_id, -maximum_prob_ood, positive_reversed= True)
        # p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-maximum_prob_id, 1-maximum_prob_ood, positive_reversed= True)

        # compute fpr95, detection_error and threshold_error 
        metrics_norm = self.compute_metrics_ood(maximum_prob_id, maximum_prob_ood)
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
            "fpr95":            float(metrics_norm['fpr95']),
            "detection_error":  float(metrics_norm['detection_error']),
            "threshold":        float(metrics_norm['thr_de'])
            
        }
        
        # save data (JSON)
        if self.name_ood_data is not None:
            path_results_folder         = self.get_path2SaveResults()
            name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
            path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
            
            print(path_result_save)
                  
            saveJson(path_file = path_result_save, data = data)
        
    def forward(self, x, thr_type = "fpr95"):
        """ discriminator forward

        Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset/CDDB scenario used as ood data, if is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            thr_type (str): choose which kind of threhsold use between "fpr95" (fpr at tpr 95%) or "avg_confidence", or "thr_de",  Default is "fpr95".
            
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
            
        if thr_type == "fpr95":
            threshold = data['fpr95']
        elif thr_type == "thr_de":
            threshold = data['threshold']
        else:                                      
            threshold = data['avg_confidence']
        
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
        pred = np.where(condition= maximum_prob < threshold, x=1, y=0)  # if true set x otherwise set y
        return pred
        
class Abnormality_module(OOD_Classifier):   # model training necessary 
    """ Custom implementation of the abnormality module using ResNet, look https://arxiv.org/abs/1610.02136 chapter 4"
    
    model_type (str): choose between: "basic", 
    
    """
    
    def __init__(self, classifier, scenario:str, model_type, useGPU = True, binary_dataset = True, batch_size = "dafault"):
        """ 
        
            scenario (str): choose between: "content", "mix", "group"
            model_type (str): choose between avaialbe model for the abnrormality module: "basic"   
            
        """
        super(Abnormality_module, self).__init__(useGPU=useGPU)
        
        # set the classifier (module A)
        self.classifier  = classifier
        self.scenario = scenario
        self.binary_dataset = binary_dataset
        self.model_type = model_type
        
        self.name               = "Abnormality_module"
        self.name_classifier    = self.classifier.classifier_name
        self.name_train         = None
        self._meta_data()
        
        # abnormality module (module B)
        self._build_model()
            
        # training parameters  
        if not batch_size == "dafault":   # default batch size is defined in the superclass 
            self.batch_size             = int(batch_size)
            
        self.lr                     = 1e-4
        self.n_epochs               = 30
        self.weight_decay           = 1e-3                  # L2 regularization term 
        
        # load data ID/OOD
        if self.binary_dataset:   # binary vs multi-class task
            self.dataset_class = CDDB_binary_Partial
        else:
            self.dataset_class = CDDB_Partial
        self._prepare_data(extended_ood = False, verbose = True)
        
        # configuration variables
        self.augment_data_train = False
        self.loss_name          = "bce"   # binary cross entropy or sigmoid cross entropy
        
        # instantiation aux elements
        self.bce     = F.binary_cross_entropy_with_logits   # performs sigmoid internally
        self.sigmoid = F.sigmoid
        self.softmax = F.softmax
        
    def _build_model(self):
        # compute shapes for the input
        x_shape = (1, *self.classifier.model.input_shape)
        x = T.rand(x_shape).to(self.device)
        probs_softmax, encoding, residual_flatten = self._forward_A(x)
        
        if self.model_type == "basic":
            self.model = Abnormality_module_Basic(probs_softmax_shape = probs_softmax.shape, encoding_shape = encoding.shape, residual_flat_shape = residual_flatten.shape)

        self.model.to(self.device)
        self.model.eval()
    
    def _meta_data(self, task_type_prog = None):
        """ prepare meta data of the current model"""
        
        if task_type_prog is None:
            if self.binary_dataset:
                task_type_prog = 0 # binary-class
            else:
                task_type_prog = 1  # multi-class 
        self.task_type_prog = task_type_prog 
        if self.scenario == "content":
            self.name_ood_data  = "CDDB_" + self.scenario + "_faces_scenario"
        else:
            self.name_ood_data  = "CDDB_" + self.scenario + "_scenario"
        
    def _prepare_data(self, extended_ood = False, verbose = False):
        """ method used to prepare Dataset class used for both training and testing
        
            ARGS:
            - extended_ood (boolean, optional): select if extend the ood data for training, using not only synthetic data
            - verbose (boolean, optional): choose to print extra information while loading the data
        """
        
        # synthesis of OOD data
        self.ood_data_train     = self.dataset_class(scenario = self.scenario, train = True,  ood = False, augment = False, label_vector= False, transform2ood = True)
        # test_dataset            = self.dataset_class(scenario = self.scenario, train = False, ood = False, augment = False, label_vector= False, transform2ood = True)
        # _ , self.ood_data_test  = sampleValidSet(trainset= self.ood_data_train, testset= test_dataset, useOnlyTest = True, verbose = True)
        
        # fetch ID data
        self.id_data_train      = self.dataset_class(scenario = self.scenario, train = True,  ood = False, augment = False, label_vector= False, transform2ood = False)
        test_dataset            = self.dataset_class(scenario = self.scenario, train = False, ood = False, augment = False, label_vector= False, transform2ood = False)
        _ , self.id_data_test   = sampleValidSet(trainset= self.id_data_train, testset= test_dataset, useOnlyTest = True, verbose = True)
                
        if verbose: print("length OOD dataset (train) synthetized -> ", len(self.ood_data_train))
        if verbose: print("length ID dataset (train and test) -> ",  len(self.id_data_train), len(self.id_data_test))
        
        # x,_ = self.ood_data_train.__getitem__(30000)
        # showImage(x)
        
        if extended_ood:
            ood_train_expansion = self.dataset_class(scenario = self.scenario, train = True,   ood = True, augment = False, label_vector= False)            
            self.ood_data_train = mergeDatasets(self.ood_data_train, ood_train_expansion) 
           
            if verbose: print("length OOD dataset after extension (just for train) -> ", len(self.ood_data_train))
        
        self.ood_data_test  = self.dataset_class(scenario = self.scenario, train = False,  ood = True, augment = False, label_vector= False)
        if verbose: print("length OOD dataset (test) -> ", len(self.ood_data_test))
        
        # train set: id data train + id data train transformed in ood (optional, + ood data train) 
        self.dataset_train = OOD_dataset(self.id_data_train, self.ood_data_train, balancing_mode="max")
        # test set: id data test + ood data test
        self.dataset_test  = OOD_dataset(self.id_data_test , self.ood_data_test,  balancing_mode="max")
        
        if verbose: print("length full dataset (train and test) with balancing -> ", len(self.dataset_train), len(self.dataset_test))
    
    def _hyperParams(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs_max": self.n_epochs,
            "weight_decay": self.weight_decay
            # "early_stopping_patience": self.patience,
            # "early_stopping_trigger": self.early_stopping_trigger,
            # "early_stopping_start_epoch": self.start_early_stopping
                }
    
    def _dataConf(self):
        
        # load not fixed config specs with try-catch

        try:
            input_shape = str(self.model.input_shape)
        except:
            input_shape = "empty"
        
        return {
            "date_training": date.today().strftime("%d-%m-%Y"),
            "model": self.model_type,
            "input_shape": input_shape,
            "data_scenario": self.scenario,
            "optimizer": self.optimizer.__class__.__name__,
            "scheduler": self.scheduler.__class__.__name__,
            "loss": self.loss_name,
            "base_augmentation": self.augment_data_train,       
            "grad_scaler": True,                # always true
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
        logits, reconstruction, encoding = self.classifier.model.forward(x)
        prob_softmax = T.nn.functional.softmax(logits, dim=1)
        if verbose: 
            print("prob shape -> ", prob_softmax.shape)
            print("encoding shape -> ",encoding.shape)
        
        # from reconstuction to residual
        residual = T.square(reconstruction - x)
        residual_flatten = T.flatten(residual, start_dim=1)
        
        if verbose: 
            print("residual shape ->", reconstruction.shape)
            print("residual (flatten) shape ->",residual_flatten.shape)
        
        return prob_softmax, encoding, residual_flatten

    def _forward_B(self, prob_softmax, encoding, residual_flatten, verbose = False):
        y = self.model.forward(prob_softmax, encoding, residual_flatten)
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
        
        prob_softmax, encoding, residual_flatten = self._forward_A(x)
        
        logit = self._forward_B(prob_softmax, encoding, residual_flatten)
        
        out   = self.sigmoid(logit)
        
        return  out
    
    def load(self, name_folder, epoch):
        
        # save folder of the train (can be used to save new files in models and results)
        self.train_name = name_folder
        
        models_path = self.get_path2SaveModels()
        self._meta_data()
        path2model              = os.path.join(models_path,  name_folder, str(epoch) + ".ckpt")
        self.modelEpochs        = epoch
        try:
            loadModel(self.model, path2model)
            self.model.eval()   # no train mode, fix dropout, batchnormalization, etc.
        except Exception as e:
            print(e)
            print("No model: {} found for the epoch: {} in the folder: {}".format(name_folder, epoch, path2model))
    
    @duration
    def train(self, additional_name = "", task_type_prog = None, test_loop = False):
        # """ requried the ood data name to recognize the task"""
        
        # 1) prepare meta-data
        self._meta_data(task_type_prog= task_type_prog)
        
        
        current_date        = date.today().strftime("%d-%m-%Y")   
        train_name          = self.name + "_" + self.model_type + "_"+ additional_name + "_" + current_date
        path_save_model     = self.get_path2SaveModels(train_name=train_name)   # specify train_name for an additional depth layer in the models file system
        path_save_results   = self.get_path2SaveResults(train_name=train_name)
        self.train_name     = train_name
        
        # 2) prepare the training components
        
        self.model.train()
        
        train_dl = DataLoader(self.dataset_train, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(train_dl)
        print("Number of steps per epoch: {}".format(n_steps))
        
        self.optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        self.scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        scaler = GradScaler()
        
        # initialize logger
        logger  = self.init_logger(path_model= path_save_model)
        
        # intialize data structure to keep track of training performance
        loss_epochs = []
        
        # learned epochs by the model initialization
        self.modelEpochs = 0
        
        # 3) learning loop
        
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # define cumulative loss for the current epoch and max/min loss
            loss_epoch = 0; max_loss_epoch = 0; min_loss_epoch = math.inf
            
            # update the last epoch for training the model
            last_epoch = epoch_idx +1
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dl), total= n_steps):
                
                # test steps loop for debug
                if test_loop and step_idx+1 == 5: break
                
                # adjust labels if cutmix has been not applied (from indices to one-hot encoding)
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(True)
                y = y.to(self.device)               # binary int encoding for each sample
                y = y.to(T.float)
                
                # take only label for the positive class (fake)
                y = y[:,1]

                # print(x.shape)
                # print(y.shape)
                
                # zeroing the gradient
                self.optimizer.zero_grad()
                
                # model forward and loss computation
                with autocast():
                    
                    
                    prob_softmax, encoding, residual_flatten = self._forward_A(x)
                                        
                    # if not basic Abnromality model, do recuction here
                    
                    logit = self._forward_B(prob_softmax, encoding, residual_flatten)
                    logit = T.squeeze(logit)
                    # print(logit.shape)
                    

                    loss = self.bce(input=logit, target= y)
                    # print(loss)
                    
                
                if loss_epoch>max_loss_epoch    : max_loss_epoch = round(loss_epoch,4)
                if loss_epoch<min_loss_epoch    : min_loss_epoch = round(loss_epoch,4)
                
                # update total loss    
                loss_epoch += loss.item()   # from tensor with single value to int and accumulation
                
                # loss backpropagation
                scaler.scale(loss).backward()
                
                # compute updates using optimizer
                scaler.step(self.optimizer)

                # update weights through scaler
                scaler.update()
                
                # lr scheduler step 
                self.scheduler.step()
                
            # compute average loss for the epoch
            avg_loss = round(loss_epoch/n_steps,4)
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
            
            # create dictionary with info frome epoch: loss + valid, and log it
            epoch_data = {"epoch": last_epoch, "avg_loss": avg_loss, "max_loss": max_loss_epoch, \
                          "min_loss": min_loss_epoch}
            logger.log(epoch_data)
            
            # test epochs loop for debug   
            if test_loop and last_epoch == 5: break
        
        
        # log GPU memory statistics during training
        logger.log_mem(T.cuda.memory_summary(device=self.device))
        
        
        # 4) Save section
        
        # save loss 
        name_loss_file          = 'loss_'+ str(last_epoch) +'.png'
        if test_loop:
            plot_loss(loss_epochs, title_plot= self.name + "_" + self.model_type, path_save = None)
        else: 
            plot_loss(loss_epochs, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_results, name_loss_file))
            plot_loss(loss_epochs, title_plot= self.name + "_" + self.model_type, path_save = os.path.join(path_save_model  ,name_loss_file), show=False)

        
        # save model
        name_model_file         = str(last_epoch) +'.ckpt'
        path_model_save         = os.path.join(path_save_model, name_model_file)  # path folder + name file
        saveModel(self.model, path_model_save)
    
        # terminate the logger
        logger.end_log()
        
    def test_risk(self, task_type_prog = None):
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
                    
                    prob_softmax, encoding, residual_flatten = self._forward_A(x)
                                        
                    # if not basic Abnromality model, do recuction here
                    
                    logit = self._forward_B(prob_softmax, encoding, residual_flatten)
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
        print(risks_id.shape)
        print(risks_ood.shape)
         
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
        p_norm_aupr, p_norm_auroc = self.compute_curves(risks_id, risks_ood)
        
        # abnormality detection
        print("Abnormality detection:")
        abnorm_base_rate = round(100*(risks_ood.shape[0]/(risks_id.shape[0] + risks_ood.shape[0])),2)
        print("\tbase rate(%): {}".format(abnorm_base_rate))
        # print("\tKL divergence (entropy)")
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(-id_entropy, -ood_entropy, positive_reversed= True)
        # kl_abnorm_aupr, kl_abnorm_auroc = self.compute_curves(1-id_entropy, 1-ood_entropy, positive_reversed= True)
        print("\tPrediction probability")
        p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-risks_id, 1-risks_ood, positive_reversed= True)
        # p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(1-risks_id, 1-risk_ood, positive_reversed= True)
        

        
        # compute fpr95, detection_error and threshold_error 
        metrics_norm = self.compute_metrics_ood(risks_id, risks_ood)
        print("OOD metrics:\n", metrics_norm)  
        metrics_abnorm = self.compute_metrics_ood(1-risks_id, 1-risks_ood, positive_reversed= True)
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
            "fpr95":            float(metrics_norm['fpr95']),
            "detection_error":  float(metrics_norm['detection_error']),
            "threshold":        float(metrics_norm['thr_de'])
            
        }
        
        # save data (JSON)
        if self.name_ood_data is not None:
            path_results_folder         = self.get_path2SaveResults(train_name=self.train_name)
            name_result_file            = 'metrics_ood_{}.json'.format(self.name_ood_data)
            path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
            
            print(path_result_save)
                
            saveJson(path_file = path_result_save, data = data)
        
    def test_threshold(self):
        self.model.eval()
        raise NotImplementedError
    
        
if __name__ == "__main__":
    #                           [Start test section] 
    
    # common classifier model definition
    classifier_name = "faces_Unet4_Scorer112p_v4_03-12-2023"
    classifier_type = "Unet4_Scorer"
    classifier_epoch = 73
    
    name_ood_data_content  = "CDDB_content_faces_scenario"
    
    # ________________________________ baseline  _______________________________________
    def test_baseline_implementation():
        ood_detector = Baseline(classifier= None, id_data_test = None, ood_data_test = None, useGPU= True)
        ood_detector.verify_implementation()
     
    def test_baseline_resnet50_CDDB_CIFAR(name_model, epoch):
        # select executions
        exe = [1,1]
        
        # [1] load deep fake classifier
        bin_classifier = DFD_BinClassifier_v1(model_type="resnet_pretrained")
        bin_classifier.load(name_model, epoch)
        
        # [2] define the id/ood data
        
        # id_data_train    = CDDB_binary(train = True, augment = False
        id_data_test     = CDDB_binary(train = False, augment = False)
        # ood_data_train   = getCIFAR100_dataset(train = True)
        ood_data_test    = getCIFAR100_dataset(train = False)
        
        # [3] define the detector
        ood_detector = Baseline(classifier=bin_classifier, task_type_prog= 0, name_ood_data="cifar100", id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
        
        # [4] launch analyzer/training
        if exe[0]: ood_detector.test_probabilties()
        
        # [5] launch testing
        if exe[1]: ood_detector.test_threshold()
    
    # TODO recompute the baseline data with cifar OOD
    
    def test_baseline_content_faces():
        classifier = DFD_BinClassifier_v4(scenario="content", model_type=classifier_type)
        classifier.load(folder_model = classifier_name, epoch = classifier_epoch)
        
        # laod id data test
        id_data_train      = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment = False, label_vector= True)
        id_data_test       = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment = False, label_vector= True)
        _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
        
        # load ood data test
        ood_data_test  = CDDB_binary_Partial(scenario = "content", train = False,  ood = True, augment = False, label_vector= False)
        
        ood_detector = Baseline(classifier=classifier, task_type_prog = 0, name_ood_data = name_ood_data_content ,  id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
        
        ood_detector.test_probabilties()
        
    
    # ________________________________ baseline + ODIN  ________________________________
    
    
    
    def test_baselineOdin_content_faces():
        classifier = DFD_BinClassifier_v4(scenario="content", model_type=classifier_type)
        classifier.load(folder_model = classifier_name, epoch = classifier_epoch)
        
        # laod id data test
        id_data_train      = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment = False, label_vector= True)
        id_data_test       = CDDB_binary_Partial(scenario = "content", train = False, ood = False, augment = False, label_vector= True)
        _ , id_data_test   = sampleValidSet(trainset = id_data_train, testset= id_data_test, useOnlyTest = True, verbose = True)
        
        # load ood data test
        ood_data_test  = CDDB_binary_Partial(scenario = "content", train = False,  ood = True, augment = False, label_vector= False)
        
        ood_detector = Baseline_ODIN(classifier=classifier, task_type_prog = 0, name_ood_data = name_ood_data_content ,  id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
        
        ood_detector.test_probabilties()
    
    # ________________________________ abnormality module  _____________________________
    
    def train_abn_basic():
        classifier = DFD_BinClassifier_v4(scenario="content", model_type=classifier_type)
        classifier.load(folder_model = classifier_name, epoch = classifier_epoch)
        
        abn = Abnormality_module(classifier, scenario="content", model_type="basic")
        abn.train(additional_name="112p")
        
        # x = T.rand((1,3,112,112)).to(abn.device)
        # y = abn.forward(x)
        # print(y)
    
    def train_abn_():pass
    
    def test_abn_content_faces(name_model, epoch):
        classifier = DFD_BinClassifier_v4(scenario="content", model_type=classifier_type)
        classifier.load(folder_model = classifier_name, epoch = classifier_epoch)
        
        abn = Abnormality_module(classifier, scenario="content", model_type="basic")
        
        abn.load(name_model, epoch)
        abn.test_risk()
    
    # test_baseline_content_faces()
    test_baselineOdin_content_faces()

    pass

    #                           [End test section] 
   
    """ 
            Past test/train launched: 
            
    test_baseline_implementation()
    test_baseline_resnet50_CDDB_CIFAR("resnet50_ImageNet_13-10-2023", 20)
    test_baseline_resnet50_CDDB_CIFAR("faces_resnet50_ImageNet_04-11-2023", 24)
    test_baseline_resnet50_CDDB_CIFAR("group_resnet50_ImageNet_05-11-2023", 26)
    test_baseline_resnet50_CDDB_CIFAR("mix_resnet50_ImageNet_05-11-2023", 21)
    
    
    
    train_abn_basic()
    test_abn_content_faces("Abnormality_module_basic_112p_05-12-2023", 30) 
    
    """
