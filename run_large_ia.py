import numpy as np

from src.win import Win23
from src.utils import Utils


if __name__ == '__main__':
    ''' Variable size IA Winograd convolution usin F(2,3)
    '''
    win_conv = Win23()
    conv_param = {'stride':1, 'pad':0}
    sparsity = 1.0

    # input activation frame size
    ia_size = 64
    # kernel size
    w_size = 3
    # output activation chunk size for F(2,3)
    oa_chunk_size = 2
    # patch overlap with previous IA patch during streaming
    f23_ovlp = 2

    # generate the test data
    ia, w, ia_tensor, w_tensor = Utils.gen_rand_sparse_vec(
                                            ia_size=ia_size,
                                            w_size=w_size,
                                            tensor_mode=True,
                                            sparsity=sparsity
                                    )

    # compute golden and winograd results
    result = np.zeros([ia_size-w_size+1, ia_size-w_size+1])

    for i in range((ia_size-w_size+1)//oa_chunk_size):
        for k in range((ia_size-w_size+1)//oa_chunk_size):
            ia_chunk = ia[f23_ovlp*i : f23_ovlp*i+4,
                          f23_ovlp*k : f23_ovlp*k+4]

            oa_chunk = win_conv.compute(ia_chunk, w)
            result[f23_ovlp*i : f23_ovlp*i+f23_ovlp,
                   f23_ovlp*k : f23_ovlp*k+f23_ovlp] = oa_chunk

    golden = Utils.gen_conv_golden(ia_tensor, w_tensor, conv_param)

    # test if the results are equal
    print('------------------------------------')
    print('Golden Result:\n{}\n'.format(golden))
    print('Winograd Result:\n{}'.format(result))
    print('------------------------------------')
    assert np.isclose(golden, result).all(), 'Results Mismatch!'
