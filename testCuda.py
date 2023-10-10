import os
import numpy as np
import torch as T


print("cuda available " + str(T.cuda.is_available()))
print("current device " + str(T.cuda.current_device()))
print("number of devices " + str(T.cuda.device_count()))

try:
    print("device 0 -> " +str(T.cuda.device(0)))
    print("name device 0 -> "+ str(T.cuda.get_device_name(0)))
except Exception as e:
    print("exception device 0 -> " +str(e))


# try:
#     print("device 1 -> " +str(T.cuda.device(1)))
#     print("name device 1 -> "+ str(T.cuda.get_device_name(1)))
# except Exception as e:
#     print("exception device 1-> " +str(e))