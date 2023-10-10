import time
import os
import argparse
import torch as T

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
		
		
def main():
	# parse main arguments
	args = parse_arguments()
	if args is None: exit()
	
	# create project folders
	create_folders(args)
	


if __name__ == "__main__":
	print("started execution {}".format(os.getcwd()+ "/main.py"))
	t_start = time.time()
	main()
	exe_time = time.time() - t_start
	print("Execution time: {} [s]".format(round(exe_time,5)))

