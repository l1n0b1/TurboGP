#-----------------------------------------------------------------------------------#
# Trimmers, as they name implies, are used to change the size of a vector or matrix #
# input in some way. For example, in image processing tasks, where GP individuals   #
# input consist of images, a trimmer may select only the border pixels of the input #
# image, while another trimmer may select the 5x5 pixels center of the image. Such  #
# examples of both trimmers are implemented here, among others that go in the same  #
# vein.                                                                             #
#
# Trimmers serve as leaf nodes for mezzanine type of primitives, in two scenarios:  #
# when in combination with high level functions, so not all subtrees necessarily    #
# convey towards high level functions (scalar inputs and constants may be used as   #
# well), or when no high level functions are being used, such that GP trees end up  #
# with single input features, scalar constants, or trimmers as leaf nodes. In this  #
# second scenario, trimmers are fundamental to allow mezzanine funtions to operate  #
# over a diverse set of input ranges, rather than operating always over a unique    #
# set of variables in all instances, i.e. they provide a richer combination of out- #
# puts that can be produced by mezzanine functions. Mezzanine and trimmer in combi- #
# nation are sometimes considered as 'feature extraction' layers in GP trees.       #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, and INAOE#
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#


import numpy as np
from Utils import *

def TFull(a):
    return a

def TNorW(a):
    q = a.reshape(int(np.sqrt(len(a))),int(np.sqrt(len(a))))
    x,_ = q.shape
    q = np.delete(q, x//2, 0)
    q = np.delete(q, x//2, 1)
    qq = blockshaped(q,x//2,x//2)
    return qq[0].flatten()

def TNorE(a):
    q = a.reshape(int(np.sqrt(len(a))),int(np.sqrt(len(a))))
    x,_ = q.shape
    q = np.delete(q, x//2, 0)
    q = np.delete(q, x//2, 1)
    qq = blockshaped(q,x//2,x//2)
    return qq[1].flatten()

def TSouW(a):
    q = a.reshape(int(np.sqrt(len(a))),int(np.sqrt(len(a))))
    x,_ = q.shape
    q = np.delete(q, x//2, 0)
    q = np.delete(q, x//2, 1)
    qq = blockshaped(q,x//2,x//2)
    return qq[2].flatten()

def TSouE(a):
    q = a.reshape(int(np.sqrt(len(a))),int(np.sqrt(len(a))))
    x,_ = q.shape
    q = np.delete(q, x//2, 0)
    q = np.delete(q, x//2, 1)
    qq = blockshaped(q,x//2,x//2)
    return qq[3].flatten()

def TCenter3(a):
    q = a.reshape(int(np.sqrt(len(a))),int(np.sqrt(len(a))))
    return crop_center(q,3,3).flatten()

def TCenter5(a):
    q = a.reshape(int(np.sqrt(len(a))),int(np.sqrt(len(a))))
    return crop_center(q,5,5).flatten()

def TOuterR(a):
    q = a.reshape(int(np.sqrt(len(a))),int(np.sqrt(len(a))))
    return np.concatenate((q[0], q[1:-1,0], q[1:-1,-1], q[-1]), axis=None)
