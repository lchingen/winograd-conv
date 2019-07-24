import numpy as np


class Win23:
    ''' Winograd convolution for F(2,3)
        + IA.shape = 4x4
        + W.shape  = 3x3
        + OA.shape = 2x2
    '''
    def __init__(self):
        self.AT = np.array([[1,1,1,0],[0,1,-1,1]])
        self.A  = self.AT.T
        self.G  = np.array([[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]])
        self.GT = self.G.T
        self.BT = np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,-1,0,1]])
        self.B  = self.BT.T
        # Debug purpose
        self.W_wino = None
        self.IA_wino = None

    def compute(self, IA, W):
        # G W GT transform
        self.W_wino = np.dot(np.dot(self.G, W), self.GT)
        # BT IA B transform
        self.IA_wino = np.dot(np.dot(self.BT, IA), self.B)
        # Winograd-domian matrix mult
        P = self.W_wino * self.IA_wino
        # AT P A inverse transform
        return np.dot(np.dot(self.AT, P), self.A)
