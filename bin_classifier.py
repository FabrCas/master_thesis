from    time                import time
from    tqdm                import tqdm
from    datetime            import date

import  torch               as T
from    torch.nn            import functional as F
from    torch.optim         import Adam, lr_scheduler
from    torch.cuda.amp      import GradScaler, autocast
from    torch.utils.data    import DataLoader

from    utilities           import *
from    dataset             import CDDB_binary, getCIFAR100_dataset,OOD_dataset
from    models              import ResNet_ImageNet, ResNet


from    sklearn.metrics     import precision_recall_curve, auc, roc_auc_score


"""
                        Binary Deepfake classifiers trained on CDDB dataset with different presets
"""


class DFD_BinClassifier(object):
    """
        binary classifier for deepfake detection using CDDB dataset
    """
    def __init__(self, useGPU = True, batch_size = 32, model_type = "resnet_pretrained"):
        """ init classifier

        Args:
            useGPU (bool, optional): flag to use CUDA device or cpu hardware by the model. Defaults to True.
        """
        super(DFD_BinClassifier, self).__init__()
        self.useGPU = useGPU
        self.batch_size = batch_size
        self.model_type = model_type
        
        
        # path 2 save
        self.path_models    = "./models/bin_class"
        self.path_results   = "./results/bin_class"

        
        # load dataset & dataloader
        self.train_dataset = CDDB_binary(train = True)
        self.test_dataset = CDDB_binary(train = False)
        
        # laod model
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        if model_type == "resnet_pretrained":
            self.model = ResNet_ImageNet(n_classes = 2).getModel()
            self.path2model_results   = os.path.join(self.path_results, "ImageNet")
        else:
            self.model = ResNet()
            self.path2model_results   = os.path.join(self.path_results, "RandomIntialization")
            
        self.modelEpochs = 0        # variable used to store the actual number of epochs used to learn the model
        self.model.to(self.device)
        
        # define loss and final activation function
        self.sigmoid = F.sigmoid
        self.bce     = F.binary_cross_entropy_with_logits
        
        # learning parameters (default)
        self.lr = 1e-5
        self.n_epochs = 20
        self.weight_decay = 0.001       # L2 regularization term 
        
    def getLayers(self, show = True):
        layers = dict(self.model.named_parameters())
        for k,v in layers.items():
            print("name: {:<30}, shape layer: {:>20}: ".format(k, str(list(v.data.shape))))
        return layers
    
    def train(self, name_train):
        
        # define train dataloader
        train_dataloader = DataLoader(self.train_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # compute number of steps for epoch
        n_steps = len(train_dataloader)
        print("Number of steps per epoch: {}".format(n_steps))
        
        # define the optimization algorithm
        optimizer =  Adam(self.model.parameters(), lr = self.lr, weight_decay =  self.weight_decay)
        
        # learning rate scheduler
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs, pct_start=0.3)
        
        # model in training mode
        self.model.train()
        
        # define the gradient scaler to avoid weigths explosion
        scaler = GradScaler()
        
        # intialize data structure to keep track of training performance
        loss_epochs = []
        
        # training loop over epochs
        for epoch_idx in range(self.n_epochs):
            print(f"\n             [Epoch {epoch_idx+1}]             \n")
            
            # define cumulative loss for the current epoch
            loss_epoch = 0
            
            # loop over steps
            for step_idx,(x,y) in tqdm(enumerate(train_dataloader), total= n_steps):
                
                # if step_idx > 5: break
                
                # prepare samples/targets batches 
                x = x.to(self.device)
                x.requires_grad_(True)
                y = y.to(self.device)               # binary int encoding for each sample
                y = y.to(T.float)
                
                # zeroing the gradient
                optimizer.zero_grad()
                
                # model forward and loss computation
                with autocast():
                    logits = self.model.forward(x)

                    loss = self.bce(input=logits, target=y)   # logits bce version
                
                # update total loss    
                loss_epoch += loss.item()
                
                # loss backpropagation
                scaler.scale(loss).backward()
                
                # compute updates using optimizer
                scaler.step(optimizer)

                # update weights through scaler
                scaler.update()
                
                # lr scheduler step 
                scheduler.step()
                
            # compute average loss for the epoch
            avg_loss = loss_epoch/n_steps
            loss_epochs.append(avg_loss)
            print("Average loss: {}".format(avg_loss))
                
            # break
        
        
        # create paths and file names for saving training outcomes
        current_date = date.today().strftime("%d-%m-%Y")
        
        path_model_folder       = os.path.join(self.path_models,  name_train + "_" + current_date)
        name_model_file         = str(self.n_epochs) +'.ckpt'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file
        
        path_results_folder     = os.path.join(self.path_results, name_train + "_" + current_date)
        name_loss_file          = 'loss_'+ str(self.n_epochs) +'.png'
        path_lossPlot_save      = os.path.join(path_results_folder, name_loss_file)
        
        # save info for the new model trained
        self.path2model_results = path_results_folder
        self.modelEpochs         = self.n_epochs
        
        # create model and results folder if not already present
        if (not os.path.exists(path_model_folder)):
            os.makedirs(path_model_folder)
        if (not os.path.exists(path_results_folder)):
            os.makedirs(path_results_folder)
        
        # save loss plot
        plot_loss(loss_epochs, title_plot= name_train, path_save = path_lossPlot_save)
        
        # save model
        saveModel(self.model, path_model_save)
    
    def test(self):
        
        # define test dataloader
        test_dataloader = DataLoader(self.test_dataset, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)

        # compute number of batches for test
        n_steps = len(test_dataloader)
        print("Number of batches for test: {}".format(n_steps))
        
        # model in evaluation mode
        self.model.eval()  
        
        # define the array to store the result
        predictions             = np.empty((0), dtype= np.int16)
        predicted_probabilities = np.empty((0), dtype= np.float32)
        targets                 = np.empty((0), dtype= np.int16)
        
        T.cuda.empty_cache()
        
        # loop over steps
        for step_idx,(x,y) in tqdm(enumerate(test_dataloader), total= n_steps):
            
            # prepare samples/targets batches 
            x = x.to(self.device)
            y = y.to(self.device)               # binary int encoding for each sample
            y = y.to(T.float)
            target = T.argmax(y, dim =-1).cpu().numpy().astype(int)
            
            with T.no_grad():
                with autocast():
                    pred, prob, _ = self.forward(x)
                    pred = pred.cpu().numpy().astype(int)
                    prob = prob.cpu().numpy()
            
            predictions             = np.append(predictions, pred, axis  =0)
            predicted_probabilities = np.append(predicted_probabilities, prob, axis  =0)
            targets                 = np.append(targets,  target, axis  =0)
            
                
        # create folder for the results
        if (not os.path.exists(self.path2model_results)):
            os.makedirs(self.path2model_results)
            
        # compute metrics from test data
        metrics_binClass(predictions, targets, predicted_probabilities, epoch_model= str(self.modelEpochs), path_save = self.path2model_results)
    
    def forward(self, x):
        """ network forward

        Args:
            x (T.Tensor): input image/images

        Returns:
            pred: label: 0 -> real, 1 -> fake
        """

        # handle single image, increasing dimensions for batch
        if len(x.shape) == 3:
            x = T.expand(1,-1,-1,-1)
            
        logits      = self.model.forward(x)
        probs       = self.sigmoid(logits)
        pred        = T.argmax(probs, -1)
        fake_prob   = probs[:,1]   # positive class probability (fake probability)
        
        return pred, fake_prob, logits
        
        
    def load(self, folder_model, epoch):
        try:
            self.path2model         = os.path.join(self.path_models,  folder_model, str(epoch) + ".ckpt")
            self.path2model_results = os.path.join(self.path_results, folder_model)
            self.modelEpochs         = epoch
            loadModel(self.model, self.path2model)
            self.model.eval()   # no train mode
        except:
            print("No model: {} found for the epoch: {} in the folder: {}".format(folder_model, epoch, self.path_models))
        
             
class OOD_Baseline(object):
    """
        Detector for OOD data
    """
    def __init__(self, classifier,  ood_data_test, ood_data_train, useGPU = True):
        """
        Class-Constructor

        Args:
            classifier (DFD_BinClassifier): classifier used for the main Deepfake detection task
            ood_data_test (torch.utils.data.Dataset): test set out of distribution
            ood_data_train (torch.utils.data.Dataset): train set out of distribution
            useGPU (bool, optional): flag to enable usage of GPU. Defaults to True.
        """
        
        # classfier used
        self.classifier  = classifier
        
        # classifier types
        self.types_classifier = ["bin_class", "multi_class", "multi_label_class"]
        
        # train
        # self.id_data_train  = CDDB_binary(train = True)
        # self.ood_data_train = ood_data_train
        # self.dataset_train  = OOD_dataset(self.id_data_train, self.ood_data_train, balancing_mode = None)
        
        # test sets
        self.id_data_test  = CDDB_binary(train = False)
        self.ood_data_test = ood_data_test
        self.dataset_test  = OOD_dataset(self.id_data_test, self.ood_data_test, balancing_mode = "max")
        
        # paths and name
        self.path_models    = "./models/ood_detection"
        self.path_results   = "./results/ood_detection"
        self.name           = "baseline"
        
        # execution specs
        self.useGPU = useGPU
        if self.useGPU: self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        else: self.device = "cpu"
        
        # learning-testing parameters
        self.batch_size     = 32
        
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
        # p_id, r_id, _ = precision_recall_curve(id_labels, prob)
        # aupr_id     = auc(r_id,p_id)
        # p_ood, r_ood, _ = precision_recall_curve(ood_labels, prob)
        # aupr_ood    =  auc(r_ood, p_ood)
        
        p, r, _ = precision_recall_curve(labels, pred)
        return  auc(r, p)
    
    def compute_auroc(self, labels, pred):
        # auroc_id    = roc_auc_score(id_labels, prob)
        # auroc_ood   = roc_auc_score(ood_labels, prob)
        return roc_auc_score(labels, pred)
        
    def compute_curves(self, id_data, ood_data, positive_reversed = False):
        # create the array with the labels initally full of zeros
        target = np.zeros((id_data.shape[0] + ood_data.shape[0]), dtype= np.int32)
        # print(target.shape)
        if positive_reversed:
            target[id_data.shape[0]:] = 1
        else:
            target[:id_data.shape[0]] = 1
        
        predictions = np.squeeze(np.vstack((id_data, ood_data)))
        
        aupr    = round(self.compute_aupr(target, predictions)*100, 2)
        auroc   = round(self.compute_auroc(target, predictions)*100, 2)
        print("\tAUROC(%)-> {}".format(auroc))
        print("\tAUPR (%)-> {}".format(aupr))
        return aupr, auroc
        
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
    
    def compute_confidence(self):     # should be not translated as direct measure of confidence
        TODO
    
    #                                     analysis and testing functions
    def analyze(self, name_classifier, task_type_prog):
        """ analyze function

        Args:
            name_classifier (nn.Module): _description_
            task_type (int): 3 possible values: 0 for binary classification, 1 for multi-class classificaiton, 2 multi-label classification
        
        """
        
        # define the dataloader 
        
        # cddbTest_dataloader = DataLoader(self.id_data_test,  batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        # ood_dataloader      = DataLoader(self.ood_data_test, batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        id_ood_dataloader   = DataLoader(self.dataset_test,  batch_size= self.batch_size, num_workers= 8, shuffle= True, pin_memory= True)
        
        # use model on selected device
        self.classifier.model.to(self.device)
        
        # define empty lsit to store outcomes
        test_logits = np.empty((0,2), dtype= np.float32)
        test_labels = np.empty((0,2), dtype= np.int32)
        
        for idx, (x,y) in tqdm(enumerate(id_ood_dataloader), total= len(id_ood_dataloader)):
            
            # to test
            if idx >= 10: break
            
            x = x.to(self.device)
            with T.no_grad():
                with autocast():
                    _ ,_, logits =self.classifier.forward(x)
                    
            # to numpy array
            logits  = logits.cpu().numpy()
            y       = y.numpy()
                
            test_logits = np.append(test_logits, logits, axis= 0)
            test_labels = np.append(test_labels, y, axis= 0)
            
        # print(test_logits.shape)
        # print(test_labels.shape)
        
        # to softmax/sigmoid probabilities
        probs = self.sigmoid(test_logits)
        
        # id/ood labels and probabilities
        id_labels  =  test_labels[:,0]
        ood_labels =  test_labels[:,1]
        prob_id     = probs[id_labels == 1]
        prob_ood    = probs[ood_labels == 1]
            
        maximum_prob_id,  entropy_id,  mean_e_id,  std_e_id  = self.compute_statsProb(prob_id)
        maximum_prob_ood, entropy_ood, mean_e_ood, std_e_ood = self.compute_statsProb(prob_ood)
        
        max_prob_id_mean = np.mean(maximum_prob_id); max_prob_id_std = np.std(maximum_prob_id)
        max_prob_ood_mean = np.mean(maximum_prob_ood); max_prob_ood_std = np.std(maximum_prob_ood)
        
        
        # in-out of distribution moments
        print("In-Distribution max prob         (mean,std) -> ", max_prob_id_mean, max_prob_id_std)
        print("Out-Of-Distribution max prob     (mean,std) -> ", max_prob_ood_mean, max_prob_ood_std)
        
        print("In-Distribution entropy          (mean,std) -> ", mean_e_id, std_e_id)
        print("Out-Of-Distribution entropy      (mean,std) -> ", mean_e_ood, std_e_ood)
        
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
        print("\tPrediction probability")
        p_abnorm_aupr, p_abnorm_auroc = self.compute_curves(-maximum_prob_id, -maximum_prob_ood, positive_reversed= True)
        
        
        # compute avg max prob
        # avg_prob_in  = np.mean(maximum_prob_id)
        # avg_prob_ood =  np.mean(maximum_prob_ood)
        
        # print(avg_prob_in)
        # print(avg_prob_ood)
    
        print("In-Distribution max prob         (mean,std) -> ", max_prob_id_mean, max_prob_id_std)
        print("Out-Of-Distribution max prob     (mean,std) -> ", max_prob_ood_mean, max_prob_ood_std)
        
        print("In-Distribution entropy          (mean,std) -> ", mean_e_id, std_e_id)
        print("Out-Of-Distribution entropy      (mean,std) -> ", mean_e_ood, std_e_ood)
        
        # store statistics in a dictionary
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
                "base_rate":    float(norm_base_rate),
                "KL_AUPR":      float(kl_norm_aupr),
                "KL_AUROC":     float(kl_norm_auroc),
                "Prob_AUPR":    float(p_norm_aupr),   
                "Prob_AUROC":   float(p_norm_auroc)
            },
            "abnormality":{
                "base_rate":    float(abnorm_base_rate),
                "KL_AUPR":      float(kl_abnorm_aupr),
                "KL_AUROC":     float(kl_abnorm_auroc),
                "Prob_AUPR":    float(p_abnorm_aupr),   
                "Prob_AUROC":   float(p_abnorm_auroc)
            }
            
        }
        
        # save data (JSON)
        name_task               = self.types_classifier[task_type_prog]
        path_ood_classifier     = os.path.join(self.path_models, name_task)
        path_ood_baseline       = os.path.join(path_ood_classifier, self.name)
        path_model_folder       = os.path.join(path_ood_baseline, name_classifier)
        name_model_file         = 'stats.json'
        path_model_save         = os.path.join(path_model_folder, name_model_file)  # path folder + name file

        # prepare file-system
        if (not os.path.exists(path_ood_classifier)):
            os.makedirs(path_ood_classifier)
        if (not os.path.exists(path_ood_baseline)):
            os.makedirs(path_ood_baseline)
        if (not os.path.exists(path_model_folder)):
            os.makedirs(path_model_folder)
        
        saveJson(path = path_model_save, data=data)
    
    def test(self, name_classifier, task_type_prog, name_odd_data = None):
        
        name_task = self.types_classifier[task_type_prog]
        path_results_ood_classifier = os.path.join(self.path_results, name_task)
        path_results_baseline       = os.path.join(path_results_ood_classifier, self.name)
        if name_odd_data is not None:
            path_results_folder         = os.path.join(path_results_baseline, name_classifier + "_" + name_odd_data)        
        else:
            path_results_folder         = os.path.join(path_results_baseline, name_classifier)    
        
        # prepare file-system
        if (not os.path.exists(path_results_ood_classifier)):
            os.makedirs(path_results_ood_classifier) 
        if (not os.path.exists(path_results_baseline)):
            os.makedirs(path_results_baseline) 
        if (not os.path.exists(path_results_folder)):
            os.makedirs(path_results_folder) 
    
    def forward(self, x):
        # handle single image, increasing dimensions for batch
        if len(x.shape) == 3:
            x = T.expand(1,-1,-1,-1)
    
    #                                     data augmentation    
    def add_noise(self, batch_input, complexity=0.5):
        return batch_input + np.random.normal(size=batch_input.shape, scale=1e-9 + complexity)

    def add_distortion_noise(self, batch_input):
        distortion = np.random.uniform(low=0.9, high=1.2)
        return batch_input + np.random.normal(size=batch_input.shape, scale=1e-9 + distortion)

        
# [test section] 
if __name__ == "__main__":
    
    # dataset = CDDB_binary()
    # test_num_workers(dataset, batch_size  =32)   # use n_workers = 8
   
    bin_classifier = DFD_BinClassifier(model_type="resnet_pretrained")
    # bin_classifier.train(name_train="resnet50_ImageNet")
    bin_classifier.load("resnet50_ImageNet_13-10-2023", 20)
    # bin_classifier.test()
    # bin_classifier.getLayers(show = True)
    
    cifar_data_train = getCIFAR100_dataset(train = True)
    cifar_data_test  = getCIFAR100_dataset(train = False)
    
    ood_detector = OOD_Baseline(classifier=bin_classifier, ood_data_train=cifar_data_train, ood_data_test= cifar_data_test, useGPU= True)
    ood_detector.analyze(name_classifier="resnet50_ImageNet_13-10-2023", task_type_prog= 0)
    
    
    
    
    
    
    
    
    