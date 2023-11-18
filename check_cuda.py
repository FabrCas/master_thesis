import os
import numpy as np
import torch as T

def isCudaAvailable():
	print("cuda available " + str(T.cuda.is_available()))
	print("current device " + str(T.cuda.current_device()))
	print("number of devices " + str(T.cuda.device_count()))

	try:
	    print("device 0 -> " +str(T.cuda.device(0)))
	    print("name device 0 -> "+ str(T.cuda.get_device_name(0)))
	except Exception as e:
	    print("exception device 0 -> " +str(e))


if __name__ == "__main__":
	isCudaAvailable()
