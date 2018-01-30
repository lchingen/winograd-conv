'''
    File: main.py
    Author: Alex lee
    Date: Jan 29 2018
'''

import numpy as np
from src.win import Win23
from src.util import *


if __name__ == '__main__':
    # Simple validation of Winograd Convolution
    win = Win23()
    conv_param = {'stride':1, 'pad':0}
    sparsity = 1.0
    
    ia, w, ia_tensor, w_tensor = rand_sparse_vec(ia_size=4, 
                                                 w_size=3, 
                                                 tensor_mode=True, 
                                                 sparsity=sparsity)
    
    golden = conv_golden(ia_tensor, w_tensor, conv_param)
    result = win.forward(ia, w)
    
    print('------------------------------------')
    print('Golden:\n{}\n'.format(golden))
    print('Winograd Result:\n{}'.format(result))
    print('------------------------------------')









