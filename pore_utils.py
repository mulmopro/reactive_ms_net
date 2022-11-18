import numpy as np

import torch

from network_tools import scale_tensor
import skfmm

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edist


def get_coarsened_list(x, scales):
    """
    X : 2D np array
    returns a list with the desired number of coarse-grained tensors
    """
    
    # converts to tensor and adds channel and batch dim
    x = torch.Tensor(add_dims(add_dims(x, 1),1))
    
    ds_x = []
    ds_x.append(x)
    
    for i in range( scales-1 ): 
        ds_x.append( scale_tensor( ds_x[-1], scale_factor=1/2 ) )
    return ds_x[::-1] # returns the reversed list (small images first)

def load_samples(feat, sample_name, scales, p_D=None, xform = None, path='.'):
    """
    feat: either mpf, edist or uz
    sample_name: sample num
    xform: data transform to perform
    """
    
    if feat == 'T':
        sample = np.load(f'{path}/{sample_name}.npy')

        
    elif feat == 'edist':
        sample = np.load(f'{path}/{sample_name}.npy')
        sample[sample!=0] = 1.
        sample = np.concatenate((sample,sample,sample),axis=1)
        sample = edist(sample)    
        sample = np.split(sample,3,axis=1)[1]  
        sample[sample==0] = 0.0


    elif feat == 'tof':
        sample = np.load(f'{path}/{sample_name}.npy')
        sample[sample!=0] = 1.
        #sample = np.concatenate((sample,sample,sample),axis=1)
        sample = edist(sample)    
        sample[sample==0]=0.0
        start = np.zeros(sample.shape)
        start[0,:]=1.0
        start[1,:]=1.0
        start=start*2.0-1.0
        sample=sample*1.0
        sample=skfmm.travel_time(start,sample)
        sample = np.split(sample,3,axis=1)[1]  
        sample = sample.max()-sample
        
        
    elif feat == 'p/D':
        sample = np.load(f'{path}/{sample_name}.npy')
        sample = np.where(sample != 0, p_D, sample)

    elif feat == 'D':
        sample = np.load(f'{path}/{sample_name}.npy')
        sample = np.where(sample != 0, D, sample)

    else:
        raise NameError('Wrong feature name or not implemented')
        
    sample_list = get_coarsened_list(sample,scales)
    return sample_list

def concat_features(feats):
    x =[torch.cat(feats,axis=1) for feats in zip(*feats)]
    return x
"""
Tensor operations
"""

def changepres(x, ttype=None):
    if ttype == 'f32':
        return x.float()
    elif ttype == 'f16':
        return x.half()

    
def torchpres(x, ttype=None):
    
    if isinstance(x,list) == True:
        x = [torchpres(xi, ttype) for xi in x]
    else:
        x = changepres(x, ttype)
    return x
    

def gpu2np(x):
    if type(x) == list:
        x = [gpu2np(xi) for xi in x]
    else:
        x = x.detach().cpu().numpy().squeeze()
    return x

def add_dims(x, num_dims):
    for dims in range(num_dims):
        x = x[np.newaxis]
    return x



def rnd_array(size, scales, device='cpu'):
    return get_coarsened_list( ((torch.rand(1,
                                            size,
                                            size,
                                            size)>0.5)*1.0).to(device),scales)
    
    
