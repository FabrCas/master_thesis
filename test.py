import torch
import numpy as np
import torch.nn.functional as F
""" 
    test sigmoid VS softmax
    
    -   sigmoid (binary class problem) gives isolated probabilities, not a probability distribution over all predicted classes
        Sigmoid outputs a single probability value for each class, making it easy to interpret. In binary classification,
        the output of a sigmoid can be directly interpreted as the probability of belonging to the positive class.
        
    -   softmax (multi class problem) instead output vector is indeed a probability distribution and that all its entries add up to 1
"""


def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
def softmax(x):
        e = np.exp(x)
        return  e/e.sum(axis=1, keepdims=True)


def sigmoid_with_softmax(logits, temperature):
    scaled_logits = logits / temperature
    
    # sigmoid_result = torch.sigmoid(scaled_logits)
    sigmoid_result = F.sigmoid(scaled_logits)
    softmax_result = F.softmax(scaled_logits, dim=1)
    # sigmoid_result = sigmoid(logits)
    # softmax_result = softmax(logits)
    
    
    return sigmoid_result, softmax_result

# logits = torch.tensor([1.0, 2.0, 3.0])

image = np.array([[1,1,1,0],[0.5,0.7,0.9,1]])

logits = torch.Tensor([[0.6, 3],[0.76,0.56]])
# logits = np.array([[0.6, 3],[0.76,0.56]])
temperature = 1.0

sigmoid_result, softmax_result = sigmoid_with_softmax(logits, temperature)
print("Sigmoid Result:", sigmoid_result)
print("Softmax Result:", softmax_result)


# # Example tensors
# tensor1 = torch.tensor([1, 2, 3])
# tensor2 = torch.tensor([4, 5, 6])

# # torch.cat example
# catenated = torch.cat((tensor1, tensor2), dim=0)
# print("torch.cat result:", catenated)

# # torch.stack example
# stacked = torch.stack((tensor1, tensor2), dim=0)
# print("torch.stack result:", stacked)