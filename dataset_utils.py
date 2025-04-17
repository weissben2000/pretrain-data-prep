import numpy as np
from copy import deepcopy
import pandas as pd

# charge_levels = [[-9e19, 400], [400, 800], [800,1200], [1200, 9e19]]

def add_noise(data, mu = 0, sig = 80):
    '''
    Add random gaussian noise to input data 
        data (np.array or pd.DataFrame): input data 
        mu (float): mean of noise 
        sig (float): standard deviation of noise
    '''
    dshape = data.shape
    noise = np.random.normal(mu, sig, dshape)
    return data + noise

def apply_threshold(x, thresh = 400):
    '''
    Apply a threshold to input data
        data (np.array or pd.DataFrame): input data 
        thresh (float): charge threshold to zero out all charge bellow
    '''
    data = deepcopy(x)
    bellowthresh = data < thresh
    data[bellowthresh] = 0
    return data

def quantize_manual(x, charge_levels, quant_values):
    '''
    Currently does nothing but BW would like to add manual quantization here as well
        data (np.array or pd.DataFrame): input data 
        charge_levels (list, shape=(N-1)): finite charge levels for boundaries of N bins. 
            eg. for N=4 bins with boundaries [-9e19, 400], [400, 800], [800,1200], [1200, 9e19]
            use: charge_levels = [400, 800, 1200]
        quan_values (list): list of values for each of N charge bins
    '''
    data = deepcopy(x)
    charge_levels= np.array(charge_levels)
    minval, maxval = [-9e19], [9e19]

    #pad the charge levels with +/- inf
    charge_levels = np.append(minval, charge_levels)
    charge_levels= np.append(charge_levels, maxval)

    #turn charge_levels into bin boundaries
    bins = None
    for c in range(len(charge_levels)-1):
        if bins is None:
            bins = [[charge_levels[c], charge_levels[c+1]]]
        else:
            bins = np.append(bins, [[charge_levels[c], charge_levels[c+1]]], axis =0)
    
    #quantize the data
    dfq = None
    for j, binbounds in enumerate(bins):
        #mask pixels by charge bin
        mask = np.float32((data.values>binbounds[0]) & (data.values<binbounds[1]))

        #set the digital value of each bin and combine bins
        if dfq is None:
            dfq = pd.DataFrame(quant_values[j]*mask)
        else:
            dfq = dfq+quant_values[j]*mask
    try:
        return dfq
    finally:
        del data, dfq, mask

def apply_offset(block, offset, pixel_array_sizeX, pixel_array_sizeY):
    '''
    Apply an offset to the per-time-stamp data from the entire pixel array
        block (list): input data at a specific time stamp (2D)
        offset (tuple): (x_offset, y_offset) to apply to the block
        pixel_array_sizeX (int): size of the pixel array in X dimension
        pixel_array_sizeY (int): size of the pixel array in Y dimension
    '''
    rows, cols = pixel_array_sizeY, pixel_array_sizeX
    new_block = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if block[i][j] != 0:
                new_i = i + offset[0]
                new_j = j + offset[1]
                if 0 <= new_i < rows and 0 <= new_j < cols:
                    new_block[new_i][new_j] = block[i][j]
    return new_block