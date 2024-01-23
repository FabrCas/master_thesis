from    time                                import time
import  random              
from    tqdm                                import tqdm
from    datetime                            import date, datetime
import  math
import  torch                               as T
import  numpy                               as np
import  os              

from    torch.nn                            import functional as F
from    torch.optim                         import Adam, lr_scheduler
from    torch.cuda.amp                      import GradScaler, autocast
from    torch.utils.data                    import DataLoader
from    torch.autograd                      import Variable
from    torchvision.transforms              import v2
from    torch.utils.data                    import default_collate

# Local imports

from    utilities                           import plot_loss, plot_valid, saveModel, metrics_binClass, loadModel, test_num_workers, sampleValidSet, \
                                            duration, check_folder, cutmix_image, showImage, image2int, ExpLogger
from    dataset                             import getScenarioSetting, CDDB_binary, CDDB_binary_Partial
# from    models                              import
from    bin_classifier                      import BinaryClassifier






if __name__ == "__main__":
    #                           [Start test section] 
    
    scenario_prog       = 1
    data_scenario       = None
    scenario_setting    = None
    
    if scenario_prog == 1: 
        data_scenario = "content"
    elif scenario_prog == 1:
        data_scenario = "group"
    
    scenario_setting = getScenarioSetting()[data_scenario]    # type of content/group
    
        
    # ________________________________ v1  ________________________________

    
    
    #                           [End test section] 
    """ 
            Past test/train launched: 
    # faces: 

    # GAN:


    
    """