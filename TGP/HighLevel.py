#-----------------------------------------------------------------------------------#
# In this file goes the implementation of all high-level functions, or primitives,  #
# that may be used. Two functions are provided as examples: the convolution and max #
# pooling operation. MaxPool requires skimage library, and since it is not usually  #
# shipped by default with most Python distributions, it is commented so the whole   #
# library can compile. Please install skimage if intend to use MaxPool.             #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, and INAOE#
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#

import numpy as np
from scipy import ndimage
#from skimage.measure import block_reduce

def Conv(a1,a2):

    a1 = a1.flatten()
    a2 = a2.flatten()

    o = a1.reshape(int(np.sqrt(len(a1))),int(np.sqrt(len(a1))))
    p = a2.reshape(int(np.sqrt(len(a2))),int(np.sqrt(len(a2))))

    # Use the smaller one as mask
    if len(a1) <= len(a2):
        return ndimage.convolve(p, o).flatten()
    else:
        return ndimage.convolve(o, p).flatten()

def MaxPool(a):

    a = a.flatten()

    if len(a) <= 16:
        # just return it
        return a
    else:
        q = np.asarray(a).reshape(int(np.sqrt(len(a))),int(np.sqrt(len(a))))
        return block_reduce(q, block_size=(2, 2), func=np.max).flatten()
