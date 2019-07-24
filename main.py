import numpy as np

from src.win import Win23
from src.utils import Utils


if __name__ == '__main__':
    win_conv = Win23()
    conv_param = {'stride':1, 'pad':0}
    sparsity = 1.0

    ia, w, ia_tensor, w_tensor = Utils.gen_rand_sparse_vec(
                                            ia_size=4,
                                            w_size=3,
                                            tensor_mode=True,
                                            sparsity=sparsity
                                    )

    golden = Utils.gen_conv_golden(ia_tensor, w_tensor, conv_param)
    result = win_conv.compute(ia, w)

    print('------------------------------------')
    print('Golden Result:\n{}\n'.format(golden))
    print('Winograd Result:\n{}'.format(result))
    print('------------------------------------')
    assert np.isclose(golden, result).all(), 'Results Mismatch!'
