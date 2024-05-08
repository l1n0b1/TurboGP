#-----------------------------------------------------------------------------------#
# Typical GP primitives such as arithmetic functions (addition, substraction, etc.) #
# trigonometric functions, etc. need to be defined here as functions that clearly   #
# specify how many parameters take as input (the arity of the primitives).          #
#
# Most of the commonly used GP primitives are already implemented here. Arithmetic  #
# (addition, substraction, multiplication and protected division), 2 Trigonometric  #
# (sine and cosine) among others (square root, IF-Then-else, etc.). Some of these,  #
# however, are prone to generate run-time errors, such as cube power (X3), x^ypower #
# (POW), because tend to overflow. Others tend to slow down individuals' evaluation #
# such as sigmoid (SIGM), so use them at your own risk.                             #
#
# Some functions commonly used as GP primitives are not defined (such as tan, log,  #
# or exp), but should be easily implemented by using the ones here as reference.    #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, and INAOE#
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#


import math
import numpy as np
#from numba import jit


#@jit(nopython=True)
def ADD(a1, a2):
    return a1 + a2

#@jit(nopython=True)
def SUB(a1, a2):
    return a1 - a2

#@jit(nopython=True)
def MUL(a1, a2):
    return a1 * a2

#@jit(nopython=True)
def DIV(a1, a2):
    if a2 != 0:
        return a1 / a2
    else:
        return 0

#@jit(nopython=True)
def SIN(a1):
    result = np.sin(a1)
    return result

#@jit(nopython=True)
def COS(a1):
    result = np.cos(a1)
    return result

#@jit(nopython=True)
def SQRT(a1):
    result = math.sqrt(math.fabs(a1))
    return result

#@jit(nopython=True)
def LOG(a1):
    if a1 == 0:
        result = 0
    else:
        result = math.log(math.fabs(a1))
    return result

#@jit(nopython=True)
def EXP(a1):
    result = np.exp(a1)
    return result

#@jit(nopython=True)
def ARCSIN(a1):
    b1 = a1
    if b1 > 1.0:
        b1 = 1.0
    if b1 < -1.0:
        b1 = -1.0
    
    result = math.asin(b1)
    return result


#@jit(nopython=True)
def X2(a1):
    return a1 ** 2

#@jit(nopython=True)
def X3(a1):
    return a1 ** 3

#@jit(nopython=True)
def MAX(a1, a2):
    return max(a1, a2)

#@jit(nopython=True)
def MEAN(a1, a2):
    return (a1 + a2)/2.0

#@jit(nopython=True)
def MIN(a1, a2):
    return min(a1, a2)

# New ANN activation functions-alike operations

#@jit(nopython=True)
def RELU(a1):
    result = max(0,a1)
    return result

#@jit(nopython=True)
def SIGM(a1):
    return 1 / (1 + math.e ** -a1)

# RARE

#@jit(nopython=True)
def IFTE(a1, a2, a3):
    if a1 > 0:
        return a2
    else:
        return a3

#@jit(nopython=True)
def POW(a1, a2):
    return np.power(math.fabs(a1),a2)
