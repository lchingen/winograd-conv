
'''
    File: util.py
    Author: Alex lee
    Date: Jan 28 2018
'''

import numpy as np

# Convolution Golden
def conv_golden(ia, w, conv_param):
    # Requires input to be tensor
    C, H, W = ia.shape
    K, C, R, S = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']

    out = np.zeros((K, 1 + (H + 2 * pad - R) / stride, 1 + (W + 2 * pad - S) / stride), dtype=np.float32)
    ia_pad = np.pad(ia, [(0,0), (pad,pad), (pad,pad)], 'constant')
    
    for f in range(K):
        for i in range(1 + (H + 2 * pad - R) / stride):
            for j in range(1 + (W + 2 * pad - S) / stride):
                ia_tmp = ia_pad[:, i * stride : i * stride + R, j * stride : j * stride + S]
                out[f, i, j] = np.sum(ia_tmp * w[f])

    return out


# Random generate vector with controlled density
def rand_sparse_vec(ia_size, w_size, tensor_mode=True, sparsity=0):
    # assume no C or K dimension
    ia = np.random.rand(ia_size, ia_size)
    w  = np.random.rand(w_size, w_size)

    # sparsify
    ia_bm = np.random.rand(np.prod(ia.shape))
    w_bm = np.random.rand(np.prod(w.shape))

    ia_bm[ia_bm >= sparsity] = 0
    w_bm[w_bm >= sparsity] = 0

    ia_bm = np.reshape(ia_bm, ia.shape)
    w_bm = np.reshape(w_bm, w.shape)
    ia[ia_bm == 0] = 0
    w[w_bm == 0] = 0

    # extend dummy C and K dimension
    ia_tensor = np.repeat(ia[np.newaxis, :], 1, axis = 0)
    w_tensor  = np.repeat(w[np.newaxis, :], 1, axis = 0)
    w_tensor  = w_tensor[np.newaxis,:]

    # tensor mode or normal mode
    if tensor_mode == True:
        return (ia, w, ia_tensor, w_tensor)
    else:
        return (ia, w)

