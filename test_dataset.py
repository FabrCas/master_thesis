import os
import sys

try:
    # Add the project root directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from dataset import * 
except:
    from dataset import * 

def test_cddbinary():
    dataset = CDDB_binary(train= True)
    # test Dataset item get
    x,y = dataset.__getitem__(0)
    print(x.shape)
    print(y)
    showImage(x, name = "CDDB sample")
    

    from torch.utils.data import DataLoader
    
    # test Dataloader from Dataset
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle= True, drop_last= True, pin_memory= True)
    show = True
    for i,(x,y) in enumerate(dataloader):
        print(x.shape)
        print(y)
        if show: 
            for i in range(x.shape[0]):
                img = x[i]
                label = y[i]
                if label[0] == 1: # real
                    name  = "real image"
                else:
                    name  = "fake image"
                showImage(img, name = name)
        break

def test_splitvalidation():
            
    from torch.utils.data import random_split
    from torch.utils.data import ConcatDataset
    
    # this implemented as a method is under the name sampleValidSet in utilties module
    
    train = CDDB_binary(train= True)
    test  = CDDB_binary(train = False)
    generator = T.Generator().manual_seed(22)
    
    print("train length", len(train))
    print("test length", len(test))
    all_data = len(train) + len(test)
    print("total samples n°", all_data)
    
    print("Data percentage distribution over sets before partition:")
    print("TrainSet [%]",round(100*(len(train)/all_data),2) )
    print("TestSet  [%]",round(100*(len(test)/all_data),2),"\n")
    
    
    if False: 
        """
            split data with the following strategy, validation set is the 10% of all data.
            These samples are extract half from training set and half from test set.
            after this we have almost the following distribution:
            training        -> 65%
            validation      -> 10%
            testing         -> 25% 
        """
        
        
        # compute relative percentage
        perc_train = round(((0.1 * all_data)*0.5/len(train)),3)
        perc_test = round(((0.1 * all_data)*0.5/len(test)),3)

        
        
        train, val_p1  = random_split(train,  [1-perc_train, perc_train], generator = generator)  #[0.92, 0.08]
        print(f"splitting train (- {perc_train}%) ->",len(val_p1), len(train))

        test,  val_p2  = random_split(test,  [1-perc_test, perc_test], generator = generator)   #[0.84, 0.16]
        print(f"splitting test (- {perc_test}%) ->",len(val_p2), len(test))
        
        val = ConcatDataset([val_p1, val_p2])
        print("validation length ->", len(val))
        
    else:
        """
            split data with the following strategy, validation set is the 10% of all data.
            These samples are extract all from the test set.
        """
        
        
        perc_test = round(((0.1 * all_data)/len(test)),3)
        
        test,  val  = random_split(test,  [1-perc_test, perc_test], generator = generator)
        print(f"splitting test (- {perc_test}%) ->",len(val), len(test))
        print("validation length", len(val))
        
    print("\nData percentage distribution over sets after partition:")
    print("TrainSet [%]",round(100*(len(train)/all_data),2) )
    print("TestSet  [%]",round(100*(len(test)/all_data),2)  )
    print("ValidSet [%]",round(100*(len(val)/all_data),2)   )
        
    
    # from subset you can easily get the dataloader in the usual way pass to dataloader
    # dl = T.utils.data.DataLoader(val1[1])
    # print(type(dl))

def test_getValid():
    train = CDDB_binary(train= True)
    test  = CDDB_binary(train= False)
    
    train, valid, test = sampleValidSet(train, test, useOnlyTest= True, verbose= True)
    print(len(train))
    print(len(valid))
    print(len(test))
    
def test_cifar():
    ds = getCIFAR100_dataset(train = False)
    train_loader = DataLoader(ds, batch_size=32, shuffle=True)
    train_loader = iter(train_loader)
    batch1 =  next(train_loader)
    batch2 = next(train_loader)
    showImage(batch1[0][31])

def test_ood():
    cddb_dataset = CDDB_binary(train = True)
    cifar_dataset = getCIFAR100_dataset(train = True)

    dataset = OOD_dataset(cddb_dataset, cifar_dataset, balancing_mode="exact", exact_samples= 50)
    print(len(dataset))
    sample = dataset[25]
    img = sample[0]
    showImage(img)
    print(sample[1])

def test_partial_bin_cddb(scenario = "content"):
    data = CDDB_binary_Partial(scenario = scenario, ood = False, train = True)
    x,y = data.__getitem__(0)
    print(x)
    showImage(x)
    
def test_partial_multi_cddb():
    # tmp = CDDB_binary(train = False)
    # data = CDDB(real_grouping= "single")
    # data = CDDB_binary_Partial(scenario="content")
    data = CDDB_Partial(scenario="content", real_grouping="categories", ood = False)
    print(data.idx2label)
    # x,y = data.__getitem__(7000)
    # print(y, data.idx2label[y])
    # showImage(x)

def test_getters(name):
    augment = False
    ood = True
    # use getters
    if name == "cifar10":
        dl_train    = getCIFAR10_dataset(train = True, augment= augment, ood_synthesis= ood)
        dl_test     = getCIFAR10_dataset(train = False, augment= augment, ood_synthesis= ood)
    elif name == "cifar100":
        dl_train    = getCIFAR100_dataset(train = True, augment= augment, ood_synthesis= ood)
        dl_test     = getCIFAR100_dataset(train = False, augment= augment, ood_synthesis= ood)
    elif name == "mnist":
        dl_train    = getMNIST_dataset(train = True)
        dl_test     = getMNIST_dataset(train = False)
    elif name == "fmnist":
        dl_train    = getFMNIST_dataset(train = True)
        dl_test     = getFMNIST_dataset(train = False)
    elif name == "svhn":
        dl_train    = getSVHN_dataset(train = True)
        dl_test     = getSVHN_dataset(train = False)
    elif name == "dtd":
        dl_train    = getDTD_dataset(train = True)
        dl_test     = getDTD_dataset(train = False)
    elif name == "tiny_imagenet":
        dl_train    = getTinyImageNet_dataset(split = "train")
        dl_test     = getTinyImageNet_dataset(split = "test") 
    else:
        print("The dataset with name {} is not available".format(name))
    
    dl_train_original  = getCIFAR10_dataset(train = True, augment= augment, ood_synthesis= False)
    
    for i in range(7):
        img, y = dl_train.__getitem__(i)
        showImage(img) 
        
        img, y = dl_train_original.__getitem__(i)
        showImage(img)
    
    print(f"train samples number: {len(dl_train)}, test samples number {len(dl_test)}")

def test_distortions():
    dt = CDDB_binary_Partial(scenario = "content", train = True,  ood = False, augment = False, label_vector= False, transform2ood = True)
    x,_ = dt.__getitem__(7)
    showImage(x, name="ood_synthesis_JPEG10_compression",save_image=True)
    # showImage(x)

if __name__ == "__main__": 
    pass