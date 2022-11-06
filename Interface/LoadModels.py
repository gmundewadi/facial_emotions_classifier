"""
LoadModels.py
Example code to load pytorch and tensorflow models 
Author: Gautam Mundewadi
"""
import torch

# Pytorch AlexNet Transfer Learn
AlexNet_TL_model = torch.load('./models/AlexNet_TL')
print("AlexNet Transfer Learned state_dict:")
for param_tensor in AlexNet_TL_model.state_dict():
    print(param_tensor, "\t", AlexNet_TL_model.state_dict()[param_tensor].size())


# TensorFlow AlextNet Train from Scratch
AlexNet_TL_model = torch.load('./models/AlexNet_Scratch')
print("AlexNet Scratch state_dict:")
for param_tensor in AlexNet_TL_model.state_dict():
    print(param_tensor, "\t", AlexNet_TL_model.state_dict()[param_tensor].size())

