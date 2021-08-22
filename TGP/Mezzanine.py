#-----------------------------------------------------------------------------------#
# Mezannine functions are defined in this file. Mezzanine functions takes as input  #
# a vector or matrix of any size and perform a operation that must return as output #
# a single scalar; e.g. the mean over a region of pixels, or the std dev of a time  #
# series, etc. In this sense, mezzanine function inputs are not strongly typed for  #
# the size of the input; they can dynamically process inputs of any size within a   #
# single GP run. This makes them fundamentally different, and more abstract, than   #
# low-level (canonical) GP primitives, which must necessarily define beforehand the #
# size/arity of its inputs.                                                         #
#
# Mezzanine functions receive their name because they may serve as the intermediate #
# step between high level nodes and low level nodes in a GP tree. They may however, #
# be used in GP individuals without high level functions and along with low-level   #
# primitives only, through the use of Trimmers; such kind of GP models can be found #
# extensively in the literature.                                                    #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, and INAOE#
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#

import numpy as np
#from numba import jit


#@jit(nopython=True)
def mMEAN(a):
    return np.mean(a)


#@jit(nopython=True)
def mSTD(a):
    return np.std(a)


#@jit(nopython=True)
def mMAX(a):
    return np.amax(a)


#@jit(nopython=True)
def mMIN(a):
    return np.amin(a)


#@jit(nopython=True)
def mMED(a):
    return np.median(a)
