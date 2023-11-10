import  os
import  torch               as T
import  numpy               as np
from    torch.nn            import functional as F
from    torch.utils.data    import DataLoader
from    torch.cuda.amp      import autocast
from    tqdm                import tqdm

from    dataset             import CDDB_binary, CDDB_binary_Partial, getCIFAR100_dataset, OOD_dataset
from    sklearn.metrics     import precision_recall_curve, auc, roc_auc_score
from    bin_classifier      import DFD_BinClassifier_v1
from    utilities           import saveJson, loadJson, metrics_binClass, metrics_OOD


types_classifier = ["bin_class", "multi_class", "multi_label_class"]

# TODO, superclass for OOD classifier
class OOD_Classifier(object):
    def __init__(self, useGPU):
        super(OOD_Classifier, self).__init__()
        
        # general paths
        self.path_models    = "./models/ood_detection"
        self.path_results   = "./results/ood_detection"
        
        # execution specs
        self.useGPU = useGPU
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        self.batch_size     = 32
        
class OOD_Baseline(OOD_Classifier):
    """
        Classifier for OOD data
    """
    def __init__(self, classifier,  id_data_test, ood_data_test, useGPU = True):
        """
        Class-Constructor

        Args:
            classifier (DFD_BinClassifier): classifier used for the main Deepfake detection task
            ood_data_test (torch.utils.data.Dataset): test set out of distribution
            ood_data_train (torch.utils.data.Dataset): train set out of distribution
            useGPU (bool, optional): flag to enable usage of GPU. Defaults to True.
        """
        super(OOD_Baseline, self).__init__(useGPU = useGPU)
        # classfier used
        self.classifier  = classifier
        
        # classifier types
        self.types_classifier = ["bin_class", "multi_class", "multi_label_class"]
        
        # train
        # self.id_data_train  = CDDB_binary(train = True)
        # self.ood_data_train = ood_data_train
        # self.dataset_train  = OOD_dataset(self.id_data_train, self.ood_data_train, balancing_mode = None)
        
        # test sets
        self.id_data_test  = id_data_test
        self.ood_data_test = ood_data_test
        try:
            self.dataset_test  = OOD_dataset(self.id_data_test, self.ood_data_test, balancing_mode = "max")
        except:
            print("Dataset data is not valid, please use instances of class torch.Dataset")
        
        # name of the classifier
        self.name           = "baseline"
        
    #                                       math aux functions
    def sigmoid(self,x):
            return 1.0 / (1.0 + np.exp(-x))
        
    def softmax(self,x):
            e = np.exp(x)
            return  e/e.sum(axis=1, keepdims=True)
            # return  e/e.sum()
        
    def entropy(self, distibution):
        """
            computes entropy from KL divergence formula, comparing the distribution with a uniform one    
        """
        return np.log(10.) + np.sum(distibution * np.log(np.abs(distibution) + 1e-11), axis=1, keepdims=True)
    
    #                                     analysis aux functions
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
        
        predictions = np.squeeze(np.vstack((id_data, ood_data)))
        
        aupr    = round(self.compute_aupr(target, predictions)*100, 2)
        auroc   = round(self.compute_auroc(target, predictions)*100, 2)
        print("\tAUROC(%)-> {}".format(auroc))
        print("\tAUPR (%)-> {}".format(aupr))
        
        return aupr, auroc
    

    def compute_metrics_ood(self, id_data, ood_data, positive_reversed = False):
        """_
            function used to compute fpr95, detection error and relative threshold.
            con be selected the positive label, the defulat one is for ID data, set to False
            with the parameter "positive_reversed" for abnormality detection (OOD).
        """
        target = np.zeros((id_data.shape[0] + ood_data.shape[0]), dtype= np.int32)
        
        if positive_reversed:
            target[id_data.shape[0]:] += 1
        else:
            target[:id_data.shape[0]] += 1
            
        print(target)
        
        predictions = np.squeeze(np.vstack((id_data, ood_data)))
        metrics_ood = metrics_OOD(targets=target, pred_probs= predictions)
        return metrics_ood 
        
    
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
    
    def compute_confidences(self, probabilities):     # should be not translated as direct measure of confidence
        """ computation of the baseline performance: maximm softmax performance (MSP)

        Args:
            probabilties (np.array): probabilities from logits

        Returns:
            confidences (np.array): list of confidences for each prediction
        """
        
        # print(probabilties.shape)
        pred_value      = np.max(probabilities, -1)
        # print(pred_value.shape)
        confidences     = np.reshape(pred_value, (len(pred_value), 1))
        # print(confidence.shape)
        
        return confidences
    
    def compute_avgConfidence(self, probabilities):
        """_
            computes the average using max probabilities from test instances.
        """
        pred_value      = np.max(probabilities, -1)   # max among class probabilities
        return np.average(pred_value)
    
    #                                     analysis and testing functions
    def analyze(self, name_classifier, task_type_prog, name_ood_data = None):
        """ analyze function, computing OOD metrics that are not threshold related

        Args:
            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset used as ood data, is is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            if None, the results will be not saved
        """
        
        # define the dataloader 
        id_ood_dataloader   = DataLoader(self.dataset_test,  batch_size= self.batch_size, num_workers= 8, shuffle= False, pin_memory= True)
        
        # use model on selected device
        self.classifier.model.to(self.device)
        
        # define empty lsit to store outcomes
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
            
        # to softmax/sigmoid probabilities
        probs = self.sigmoid(pred_logits)
        
        
        # separation of id/ood labels and probabilities
        id_labels  =  dl_labels[:,0]                # filter by label column
        ood_labels =  dl_labels[:,1]
        prob_id     = probs[id_labels == 1]         # split forward probabilites between ID adn OOD, still a list of probabilities for each class learned by the model
        prob_ood    = probs[ood_labels == 1]
        
        # compute confidence (all)
        conf_all = round(self.compute_avgConfidence(probs),3)
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
        if name_ood_data is not None:            
            name_task = self.types_classifier[task_type_prog]
            path_results_ood_classifier = os.path.join(self.path_results, name_task)
            path_results_baseline       = os.path.join(path_results_ood_classifier, self.name)
            path_results_folder         = os.path.join(path_results_baseline, name_classifier)    
            name_result_file            = 'metrics_ood_{}.json'.format(name_ood_data)
            path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
            
            print(path_result_save)
            
            # prepare file-system
            if (not os.path.exists(path_results_ood_classifier)):
                os.makedirs(path_results_ood_classifier) 
            if (not os.path.exists(path_results_baseline)):
                os.makedirs(path_results_baseline) 
            if (not os.path.exists(path_results_folder)):
                os.makedirs(path_results_folder) 
            
            saveJson(path = path_result_save, data = data)
    
    def test_implementation(self):
        """
            Function used to verify the correct implementation of the baseline, re-creating the original experiment of the paper
        """
        print("ciao")
        
    
    def testing_binary_class(self, name_classifier, task_type_prog, name_ood_data, thr_type = "fpr95"):
        """
            This test compute metrics (binary classification ID/OOD) that are threshold related (discriminator)
            
                Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset used as ood data, is is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            if None, the results will be not saved
            thr_type (str): choose which kind of threhsold use between "fpr95" (fpr at tpr 95%) or "avg_confidence", or "thr_de",  Default is "fpr95".
            
        """
        
        # load data from analyze
        name_task                   = self.types_classifier[task_type_prog]
        path_results_ood_bin        = os.path.join(self.path_results, name_task)
        path_results_baseline       = os.path.join(path_results_ood_bin, self.name)
        path_results_folder         = os.path.join(path_results_baseline, name_classifier)    
        name_result_file            = 'metrics_ood_{}.json'.format(name_ood_data)
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
        
        
        probs = self.sigmoid(test_logits)
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
        
        # print(pred)
        # print(target)
    
        # prepare file-system
        if (not os.path.exists(path_results_ood_bin)):
            os.makedirs(path_results_ood_bin) 
        if (not os.path.exists(path_results_baseline)):
            os.makedirs(path_results_baseline) 
        if (not os.path.exists(path_results_folder)):
            os.makedirs(path_results_folder)
        
        # compute and save metrics 
        name_resultClass_file  = 'metrics_ood_classification_{}.json'.format(name_ood_data)
        metrics_class =  metrics_binClass(preds = pred, targets= target, pred_probs = None, path_save = path_results_folder, name_ood_file = name_resultClass_file)
        
        print(metrics_class)

    def forward(self, x, name_classifier, task_type_prog, name_ood_data, thr_type = "fpr95"):
        """ discriminator forward

        Args:
            x (torch.Tensor): input image to be discriminated (real/fake)

            name_classifier (str): deepfake detection model used (name from models folder)
            task_type_prog (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
            name_ood_data (str): name of the dataset used as ood data, is is a partition of CDDB specify the scenario: "content","group","mix", Default is None. 
            if None, the results will be not saved
            thr_type (str): choose which kind of threhsold use between "fpr95" (fpr at tpr 95%) or "avg_confidence", or "thr_de",  Default is "fpr95".
            
        """        
        if not(isinstance(x, T.Tensor)):
            x = T.tensor(x)
            
        # adjust to handle single image, increasing dimensions for batch
        if len(x.shape) == 3:
            x = T.expand(1,-1,-1,-1)
        
        # load the threshold
        # load data from analyze
        name_task                   = self.types_classifier[task_type_prog]
        path_results_ood_bin        = os.path.join(self.path_results, name_task)
        path_results_baseline       = os.path.join(path_results_ood_bin, self.name)
        path_results_folder         = os.path.join(path_results_baseline, name_classifier)    
        name_result_file            = 'metrics_ood_{}.json'.format(name_ood_data)
        path_result_save            = os.path.join(path_results_folder, name_result_file)   # full path to json file
        
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
        probs = self.sigmoid(logits)
        maximum_prob = np.max(probs, axis=1)
        
        # apply binary threshold
        pred = np.where(condition= maximum_prob < threshold, x=1, y=0)
        return pred
    
if __name__ == "__main__":
    
    
    def test_baseline_implementation():
        ood_detector = OOD_Baseline(classifier= None, id_data_test = None, ood_data_test = None, useGPU= True)
        ood_detector.test_implementation()
     
    def baseline_resnet50_CDDB_CIFAR(name_model, epoch):
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
        ood_detector = OOD_Baseline(classifier=bin_classifier, id_data_test = id_data_test, ood_data_test = ood_data_test, useGPU= True)
        
        # [4] launch analyzer/training
        if exe[0]: ood_detector.analyze(name_classifier=name_model, task_type_prog= 0, name_ood_data="cifar100")
        
        # [5] launch testing
        if exe[1]: ood_detector.testing_binary_class(name_classifier=name_model, task_type_prog = 0, name_ood_data="cifar100", thr_type= "")
        
    test_baseline_implementation()
    

# baseline_resnet50_CDDB_CIFAR("faces_resnet50_ImageNet_04-11-2023", 24)
# baseline_resnet50_CDDB_CIFAR("groups_resnet50_ImageNet_05-11-2023", 26)
# baseline_resnet50_CDDB_CIFAR("mix_resnet50_ImageNet_05-11-2023", 21)

