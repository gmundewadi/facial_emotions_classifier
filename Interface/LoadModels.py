"""
LoadModels.py
Example code to load pytorch and tensorflow models 
Author: Gautam Mundewadi
"""
import torch

AlexNet_TL_model = torch.load('../models/AlexNet_Scratch_dataAugment')
for param_tensor in AlexNet_TL_model.state_dict():
    print(param_tensor, "\t", AlexNet_TL_model.state_dict()[param_tensor].size())

