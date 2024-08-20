import argparse
import torch
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel, LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn
from typing import List
import math
from tqdm import tqdm
import os




class latent_guidance_predictor(nn.Module):
    def __init__(self, output_dim, input_dim, num_encodings):
        super(latent_guidance_predictor, self).__init__()
        self.num_encodings = num_encodings
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(512, 256),         
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),     
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),      
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, t):
        # Concatenate input pixels with noise level t and positional encodings
        pos_encoding = [torch.sin(2 * math.pi * t * (2 **-l)) for l in range(self.num_encodings)]
        pos_encoding = torch.cat(pos_encoding, dim=-1)
        x = torch.cat((x, t, pos_encoding), dim=-1)
        x = x.flatten(start_dim=0, end_dim=2)
        
        return self.layers(x)

#add to inpainting_st

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.float() for f in features if f is not None and isinstance(f, torch.Tensor)]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.float())

def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out

def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out


#Add to the ddim_sample_p

    
def resize_and_concatenate(activations: List[torch.Tensor], reference):
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = reference.shape[2:]
    resized_activations = []
    for acts in activations:
        acts = nn.functional.interpolate(
            acts, size=size, mode="bilinear"
        )
        acts = acts[:1]
        acts = acts.transpose(1,3)
        resized_activations.append(acts)
    
    return torch.cat(resized_activations, dim=3)

def extract_features(sampler,latent_image, blocks, text_embeddings, timesteps):
    ###reduce memory consum
    h_in=[]#for remove the hook
    h_out=[]
    h_m=[]

    latent_model_input = torch.cat([latent_image] * 2)
    timesteps = torch.cat([timesteps] * 2)

    activations = []
    save_hook = save_out_hook
    feature_blocks = []
    for idx, block in enumerate(sampler.model.model.diffusion_model.input_blocks):
        if idx in blocks:
            h = block.register_forward_hook(save_hook)
            h_in.append(h)
            feature_blocks.append(block) # each block contains 1 layer


    for idx, block in enumerate(sampler.model.model.diffusion_model.output_blocks):
        if idx in blocks:
            h = block.register_forward_hook(save_hook)
            h_out.append(h)
            feature_blocks.append(block)  # each block contains 2 layers

    #mid_block    
    block = sampler.model.model.diffusion_model.middle_block
    h_m = block.register_forward_hook(save_hook)
    feature_blocks.append(block)
    
    #print(latent_model_input.shape,timesteps.shape)###
    #print(type(text_embeddings),text_embeddings.shape)
    #print(text_embeddings['c_concat'][0].shape,text_embeddings['c_crossattn'][0].shape)##


    with torch.no_grad():
        noise_pred = sampler.model.apply_model(latent_model_input, timesteps, text_embeddings)#
    

    # Extract activations
    for block in feature_blocks:
        activations.append(block.activations)
        block.activations = None

    activations = [activations[0], activations[1], activations[2], activations[3], activations[4], activations[5], activations[6]]

    for i in range(3):#remove the hook
        h_in[i].remove()
        h_out[i].remove()
    h_m.remove()
    return activations

#add to the inpainting_st