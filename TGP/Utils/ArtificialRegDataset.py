#-----------------------------------------------------------------------------------#
# This file provides functions to generate artificial regression datasets. It aims  #
# to be a comprehensive collection of artificial regression datasets proposed in GP #
# literature over the course of the years, and commonly re-used by succesive works. #
#
# References will be cited when applicable. Despite an exhaustive bibliography rese-#
# arch, some references might be missing or wrong. Feel free to modify accordinly.  #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, et. al. 2020.                                            #
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from inspect import signature


def polynomial(x):
    ''' Standard polynomial function; probably originally proposed by Koza on some early
    book or paper, as a testbed for the just recently  introduced GP framework, but since 
    then appearing all over the literature. Some authors that use it as benchmark:
    '''
    
    y = (x**4) - (x**3) + (x**2) - x
    
    return y

def poly_float(x):
    ''' This is a slightly modified version of the, all-time clasic, polynomial function.
    This version changes the integer coefficients for floats smaller than one. It is also 
    one degree lesser than the original one, but it has an bias coefficient.
    
    Proposed by Evett & Fdez (1998) as a benchmark for their proposed "Numeric Mutation"
    genetic operation.'''
    
    y = (x**3) - (0.3*(x**2)) - (0.4*x) - 0.6
    
    return y

def keijzer2(x):
    ''' Marteen Keijzer (2003) proposed one of the most widely used GP artificial problems
    sets for regression benchmark. The original aim of his proposed set of problems was to 
    test the use of interval arithmetic and linear scaling in GP. In the first two problems,
    keijzer2 and keijzer3, it is evident how GP can benefit from his proposed linear scaling
    method.'''
    
    y = x**2
    
    return y

def keijzer3(x):
    ''' Slightly modified version of Keijzer test problem for linear scaling '''
    
    y = (3.1416*(x**2)) + 100
    
    return y

def keijzer4(x):
    ''' The first original artificial problem from Keijzer's compilation, as the first two
    above, keijzer2 and keijzer3 are used as a preamble in his paper (Keijzer, 2003) and 
    are not numbered. '''
    
    y = 0.3 * x * np.sin(2 * 3.14159 * x)
    
    return y

def keijzer5(x):
    ''' Keijzer Eq. 5. Originally appearing in Salustowicz & Schmidhuber (1997).'''
    
    y = (x**3) * np.exp(-x) * np.cos(x) * np.sin(x) * ((np.sin(x)**2) * np.cos(x) - 1)
    
    return y

def keijzer6(x,y,z):
    ''' Keijzer Eq. 6. Keijzer, (2003).'''
    
    f = (30.0 * x * z) / ((x - 10.0) * (y**2))
    
    return f

def keijzer7(x):
    ''' Keijzer Eq. 7. Originally taken from Streeter & Becker (2003).'''
    
    f = 0
    
    for i in range(1,int(x)):
        f += (1. / i)
    
    return f

def keijzer8(x):
    ''' Keijzer Eq. 8. Originally taken from Streeter & Becker (2003).'''
    
    y = np.log(x)
    
    return y

def keijzer9(x):
    ''' Keijzer Eq. 9. Originally taken from Streeter & Becker (2003).'''
    
    y = np.sqrt(x)
    
    return y

def keijzer10(x):
    ''' Keijzer Eq. 10. Originally taken from Streeter & Becker (2003).'''
    
    y = np.arcsinh(x)
    
    return y

def keijzer11(x,y):
    ''' Keijzer Eq. 11. Originally taken from Streeter & Becker (2003).'''
    
    z = x**y
    
    return z

def keijzer12(x,y):
    ''' Keijzer Eq. 12. Originally taken from Topchy & Punch (2001).'''
    
    z = (x*y) + np.sin((x - 1)*(y - 1))
    
    return z

def keijzer13(x,y):
    ''' Keijzer Eq. 13. Originally taken from Topchy & Punch (2001).'''
    
    z = (x**4) - (x**3) + ((y**2)/2) - y
    
    return z

def keijzer14(x,y):
    ''' Keijzer Eq. 14. Originally taken from Topchy & Punch (2001).'''
    
    z = 6.0 * np.sin(x) * np.cos(y)
    
    return z

def keijzer15(x,y):
    ''' Keijzer Eq. 15. Originally taken from Topchy & Punch (2001).'''
    
    z = 8.0 / (2.0 + (x**2) + (y**2))
    
    return z

def keijzer16(x,y):
    ''' Keijzer Eq. 16. Originally taken from Topchy & Punch (2001).'''
    
    z = ((x**3)/5) + ((y**3)/2) - y - x
    
    return z

def Nguyen1(x):
    ''' "Nguyen" is another popular collection of artificial regression toy problems
    commonly used in GP works for benchmarking purposes. Originally proposed by Uy et al (2011).
    This is the first of such functions.'''
    
    y = (x**3) + (x**2) + x
    
    return y

def Nguyen2(x):
    ''' Nguyen-2 (Uy et al., 2011)'''
    
    y = (x**4) + (x**3) + (x**2) + x
    
    return y

def Nguyen3(x):
    ''' Nguyen-3 (Uy et al., 2011)'''
    
    y = (x**5) + (x**4) + (x**3) + (x**2) + x
    
    return y

def Nguyen4(x):
    ''' Nguyen-4 (Uy et al., 2011)'''
    
    y = (x**6) + (x**5) + (x**4) + (x**3) + (x**2) + x
    
    return y

def Nguyen5(x):
    ''' Nguyen-5 (Uy et al., 2011)'''
    
    y = (np.sin(x**2)*np.cos(x)) - 1.0
    
    return y

def Nguyen6(x):
    ''' Nguyen-6 (Uy et al., 2011)'''
    
    y = np.sin(x) + np.sin(x + x**2)
    
    return y

def Nguyen7(x):
    ''' Nguyen-7 (Uy et al., 2011)'''
    
    y = np.log(x + 1) + np.log((x**2) + 1)
    
    return y

def linobi0(x):
    ''' Similar to keijzer9, this simple, but non-linear, function can serve as a
    quick GP benchmark, although only when sin is not included in the primitives set;
    otherwise it would be trivial for GP to solve.'''
    
    y = np.sin(x)
    
    return y

def linobi1(x,y):
    ''' This function is taken from matplotlib documentation for 3D surfaces ploting: 
    https://matplotlib.org/stable/gallery/mplot3d/surface3d.html. I was originally 
    following the example in order to add a 3D plotting method to this module, but then 
    I thought it would be nice to use it as GP synthetic benchmark as well :) '''
    
    z = np.sin(np.sqrt(x**2 + y**2))
    
    return z

def linobi2(x,y):
    ''' This function is taken from gplearn documentation, 
    https://gplearn.readthedocs.io/en/stable/examples.html It was proposed as a good
    artificial regression dataset that serves as an example of a function difficult to
    approximate by decision trees and random forests, while being relatively easy for GP.
    Therefore, I also decided to include it on this collection of benchmark problems, as
    it can serve as an unit test for GP in overall.'''
    
    z = x**2 - y**2 + y - 1
    
    return z

def linobi3(x,y):
    ''' This function is taken from Morse & Stanley (2016). In their paper, they proposed
    this function as testbed for a evolutionary method for neural net weights optimization.
    This function is relatively easy for neural networks to approximate; on the other hand,
    it is particularly difficultt for GP, specifically when trigonometric functions are not
    part of the primitives set. Therefore, I considered it fit for benchmarking GP against
    neural nets, as well as research into automatic subroutine discovery. '''
    
    z = (np.sin(5*x*(3*y + 1)) + 1.) / 2.
    
    return z



def generator(func, train_range, train_samples, test_samples, mode='random', test_mode=None, test_range=None, save_to_disk=None, online=False, batch_size=100):
    '''
    func (method): python method that defines the math function from which the dataset will be generated.
    
    train_range (tuple): range in the form of tuple (a,b) that defines the bounds of the area to be sampled, with 
    lower bound a, and upper bound b. When dealing with two or higher dimensional functions, these bounds operate
    symmetrically across all dimensions/axis.
    
    train_samples (int): number of training samples to generate.
    
    test_samples (int): number of testing samples to generate.
    
    mode: ('random' or 'l_mesh'; default 'random'): defines the way samples will be generated; with 'random' all
    input vectors for the test function will be randomly generated with a uniform distribution; with 'l_mesh' 
    input values will be generated equally spaced (given the number of samples to be generated) (using numpy linspace)
    across all axis (i.e. linearly for 1-dimensional functions, mesh for 2 and 3 dimensional functions, etc.)
    
    test_mode ('random', 'l_mesh' or None; default None): same as 'mode', but for the test samples. If None, then
    will use the value of 'mode'.
    
    test_range (tuple): same as 'train_range', but for the testing samples. Similarly to 'test_mode', this option
    is used if the testing set is to be generated with a different pattern from that of the training set. If not
    defined then will use the same bounds as in train_range.
    
    save_to_disk (string): file name where the generated dataset will be saved, using TurboGP pickle convention, 
    If not defined (i.e., None), then no file will be generated and only a dict containing the dataset will be 
    returned by this method.
    
    online (boolean; default False): specifies if the generated dataset should be stored in the form of minibatches,
    i.e. for online learning, or in a single array.   
    
    batch_size (int; default 100): size of the minibatches if online = true, ignored otherwise.
    
    
    Returns: a dict contained the generated dataset in following format:
            - 'x_training': input training samples
            - 'y_training': output training labels
            - 'x_testing': input testing samples
            - 'y_testing': output testing labels.
            
            x_training and y_training get replaced by 'batchesX' and 'batchesY' when online = True. 
            batchesX is a partition of 'x_training' composed of sets of size batch_size.
    
    '''
    
    rng = np.random.default_rng()
    
    if test_mode is None:
        test_mode = mode
    
    if test_range is None:
        test_range = train_range
        
    func_dim = len(signature(func).parameters)
    
    
    # training data
    
    a, b = train_range
    
    if mode == 'random':
        
        x_training = (b - a) * rng.random((train_samples, func_dim)) + a
    
    elif mode == 'l_mesh':
        
        if func_dim == 1:
            
            x_training = np.linspace(a, b, train_samples).reshape(train_samples,1)
        
        elif func_dim == 2:
            
            dim_len = int(np.sqrt(train_samples))
            
            x = np.linspace(a,b,dim_len)
            y = np.linspace(a,b,dim_len)
            
            l = []
            for i in range(dim_len):
                for j in range(dim_len):
                    sample = [x[i], y[j]]
                    l.append(sample)
            
            x_training = np.asarray(l)
        
        elif func_dim == 3:
            
            dim_len = int(np.cbrt(train_samples))
            
            x = np.linspace(a,b,dim_len)
            y = np.linspace(a,b,dim_len)
            z = np.linspace(a,b,dim_len)
            
            l = []
            for i in range(dim_len):
                for j in range(dim_len):
                    for k in range(dim_len):
                        sample = [x[i], y[j], z[k]]
                        l.append(sample)
            
            x_training = np.asarray(l)
        
        elif func_dim == 4:
            
            dim_len = int(np.sqrt(np.sqrt(train_samples)))
            
            x = np.linspace(a,b,dim_len)
            y = np.linspace(a,b,dim_len)
            z = np.linspace(a,b,dim_len)
            w = np.linspace(a,b,dim_len)
            
            l = []
            for i in range(dim_len):
                for j in range(dim_len):
                    for k in range(dim_len):
                        for m in range(dim_len):
                            sample = [x[i], y[j], z[k], w[m]]
                            l.append(sample)
            
            x_training = np.asarray(l)
        
        else:
            print("Mesh mode implemented only up to 4-dimensional functions")
            return None
    
    else:
        print("Error: Unknown train data mode selected")
        return None
    
    y_training = []
    for sample in x_training:
        y_training.append(func(*list(sample)))
    
    y_training = np.asarray(y_training)#.reshape(train_samples,1)
    
    
    # testing data
    
    a, b = test_range
    
    if test_mode == 'random':
        
        x_testing = (b - a) * rng.random((test_samples, func_dim)) + a
    
    elif test_mode == 'l_mesh':
        
        if func_dim == 1:
            
            x_testing = np.linspace(a, b, test_samples).reshape(test_samples,1)
        
        elif func_dim == 2:
            
            dim_len = int(np.sqrt(test_samples))
            
            x = np.linspace(a,b,dim_len)
            y = np.linspace(a,b,dim_len)
            
            l = []
            for i in range(dim_len):
                for j in range(dim_len):
                    sample = [x[i], y[j]]
                    l.append(sample)
            
            x_testing = np.asarray(l)
        
        elif func_dim == 3:
            
            dim_len = int(np.cbrt(test_samples))
            
            x = np.linspace(a,b,dim_len)
            y = np.linspace(a,b,dim_len)
            z = np.linspace(a,b,dim_len)
            
            l = []
            for i in range(dim_len):
                for j in range(dim_len):
                    for k in range(dim_len):
                        sample = [x[i], y[j], z[k]]
                        l.append(sample)
            
            x_testing = np.asarray(l)
        
        elif func_dim == 4:
            
            dim_len = int(np.sqrt(np.sqrt(test_samples)))
            
            x = np.linspace(a,b,dim_len)
            y = np.linspace(a,b,dim_len)
            z = np.linspace(a,b,dim_len)
            w = np.linspace(a,b,dim_len)
            
            l = []
            for i in range(dim_len):
                for j in range(dim_len):
                    for k in range(dim_len):
                        for m in range(dim_len):
                            sample = [x[i], y[j], z[k], w[m]]
                            l.append(sample)
            
            x_testing = np.asarray(l)
        
        else:
            print("Mesh mode implemented only up to 4-dimensional functions")
            return None
    
    else:
        print("Error: Unknown test data mode")
        return None
    
    y_testing = []
    for sample in x_testing:
        y_testing.append(func(*list(sample)))
    
    y_testing = np.asarray(y_testing)#.reshape(test_samples,1)
    
        
    # store dataset to TurboGP dict format
    
    if online:
        no_batches = train_samples // batch_size
        
        batchesX = x_training.reshape(no_batches, batch_size, func_dim)
        batchesY = y_training.reshape(no_batches, batch_size)
        
        ds_ = {'batchesX': batchesX,
               'batchesY': batchesY,
               'x_testing': x_testing,
               'y_testing': y_testing}
    
    else:
        
        ds_ = {'x_training': x_training,
               'y_training': y_training,
               'x_testing': x_testing,
               'y_testing': y_testing}
        
    if save_to_disk is not None:
        pickle.dump(ds_, open(save_to_disk, "wb"), protocol=2 )
        
    
    return ds_ 


def plot2d(func, plot_range , resolution):
    
    ds_ = generator(func, plot_range, resolution, 10, mode='l_mesh')
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.grid(True)
    plt.axhline(color='black', lw=1.)
    plt.axvline(color='black', lw=1.)

    line1, = ax.plot(ds_['x_training'], ds_['y_training'], 'b-', label=func.__name__)

    ax.legend()
    plt.show()
    
    return None


def plot3D(func, plot_range, resolution):
    
    ds_ = generator(func, plot_range, resolution, 10, mode='l_mesh')
    
    a,b = plot_range
    ax_res = int(np.sqrt(resolution))
    
    X = np.linspace(a, b, ax_res)
    Y = np.linspace(a, b, ax_res)
    
    X, Y = np.meshgrid(X, Y)
    
    Z = ds_['y_training'].reshape(ax_res,ax_res)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,10))
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.7)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    

    plt.show()
    
    return None


if __name__ == "__main__":
    
    # Generate a dataset composed of 1000 training samples split in 10 minibatches of 100 samples each, randomly sampling "linobi0" function, 
    # with input values from -3.14 to + 3.14, with 100 testing sample, save it to pickle file "linobi0-pi-1000-100.npz"
    
    ds_ = generator(func=linobi0, train_range=(-3.14,3.14), train_samples=1000, test_samples=100, mode='random', save_to_disk="linobi0-pi-1000-100.npz", online=True, batch_size=100)
    
    # refer to file quick_run.py to see a quick GP run using generated dataset above

'''

Bibliography

Keijzer, M. (2003, April). Improving symbolic regression with interval arithmetic and linear scaling. In European Conference on Genetic Programming (pp. 70-82). Springer, Berlin, Heidelberg.

Salustowicz, R., & Schmidhuber, J. (1997). Probabilistic incremental program evolution. Evolutionary computation, 5(2), 123-141.

Streeter, M., & Becker, L. A. (2003). Automated discovery of numerical approximation formulae via genetic programming. Genetic Programming and Evolvable Machines, 4(3), 255-286.

Topchy, A., & Punch, W. F. (2001, July). Faster genetic programming based on local gradient search of numeric leaf values. In Proceedings of the genetic and evolutionary computation conference (GECCO-2001) (Vol. 155162). Morgan Kaufmann.

Uy, N. Q., Hoai, N. X., O’Neill, M., McKay, R. I., & Galván-López, E. (2011). Semantically-based crossover in genetic programming: application to real-valued symbolic regression. Genetic Programming and Evolvable Machines, 12(2), 91-119.

Morse, G., & Stanley, K. O. (2016, July). Simple evolutionary optimization can rival stochastic gradient descent in neural networks. In Proceedings of the Genetic and Evolutionary Computation Conference 2016 (pp. 477-484).

'''
