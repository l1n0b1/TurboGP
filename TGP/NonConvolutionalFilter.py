#-----------------------------------------------------------------------------------#
# This file implements a GP-individual based on a single tree and aimed to perform  #
# image denoising. It is fundamentally the same as the SimpleRegresor individual,   #
# with the difference that instead of receiving as label a single scalar, it recei- #
# ves a matrix the same size as the input sample. That is, a training set for this  #
# type of individuals must consist in noisy-clean images patches pairs (or noisy -  #
# noise_mask pairs, if a residual approach is taken). From there, individuals can   #
# take two behaviors: (a) they be can slided across the entire input sample, in a   #
# convolutional fashion, in order to clean the full input image patch, or (b) they  #
# may predict the correct value/noise level of only the central pixel (non-convolu- #
# tional approach). It is recommended that individuals are trained using (b) with   #
# training sets composed of small image batches (9x9 up to 21x21), and then can be  #
# tested or deployed in field using the convolution-alike prediction method.        #
#
# This python class exemplifies a custom type of GP individual, developed from the  #
# SimpleRegresor class, used as reference, and then modifiying it to perform a more #
# specialized task.                                                                 #
#
# Similarly to SimpleRegresor, this GP Individual class is thought with only low-   #
# level primitives in mind, i.e., GP operations defined here, are not configured to #
# support mezzanine or high level primitives (in regard to crossover operations, as #
# mutation supports all level of primitives by default). Please refert to classes   #
# defined in NonConvolutionalMezzanineFilter.py and NonConvolutionalHighFilter.py   #
# to find individuals with multiple-abstraction level crossover support, and perform#
# a comparison with the one defined here.                                           #
#
# For a complete guide on the usage of this GPIndividual class, and the TurboGP     #
# library as a whole, see the jupyter notebooks provided as examples, LowGP Filter, #
# MidGP Filter and HighGP Filter.                                                   #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, and INAOE#
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#



from GPIndividuals import *
from GPOperators import *

import scipy.ndimage as ndimage

import numpy as np

import random


class NonConvFilter:
    ''' This is a Non-Convolutional Filter. It is a filter that is trained with very small mini-patches, say 9x9, 15x15
    or up to 21x21. Since the the patches are so small, the filter does not gets convoluted across them, it simply acts
    as a regressor with vectorial input that attempts to reconstruct the central pixel of the given mini-patch (scalar
    output). But it can also perform convolution-alike image processing, for testing purposes on full sized images. '''

    def __init__(self, lateral_window_size, complexity, grow_method='variable'):
        ''' This is the constructor for the Non-Convolutional filter class. Requires as input the lateral sizes of the
        patches to process (e.g. 7x7 patches, lateral_window_size=7), the max tree depth (complexity), and (optionally)
        the tree grow method.'''

        self.input_window = lateral_window_size

        # The input set is calculated from squaring the window size, e.g. 7x7 patches, 49 pixels, hence 1 to 49 features.
        input_set = np.arange(self.input_window * self.input_window)

        self.tree = Tree(max_tree_depth = complexity,
                         i1_set = input_set,
                         grow_method = grow_method)

        self.tree.grow_random_tree()

        self.fitness_value = float('Inf') # because it is a minimization problem

    def nc_predict(self, instance):
        ''' This is the predict function used for training purposes. It does not slide the tree over the input.'''

        # Simple regressor
        result = self.tree.evaluate(instance)

        return result

    def nc_case_fitness(self, instance, label):
        ''' This is the fitness evaluation function for a single instance that is used when training. It uses nc_predict
        function to attempt to perform a prediction given an input instance and then compares against the correct value
        by looking at the center pixel from the passed patch-label. The prediction can be either the correct pixel value
        (denoised) or the noise level (residual approach), this will depend on the kind of labels passed for training.'''

        # Perform a prediction
        output = self.nc_predict(instance.flatten())

        # extract central pixel from label
        Pos = self.input_window // 2
        Pixel = label[Pos,Pos]

        # Square error
        result =  ((Pixel - output)**2)

        return result

    def fitness(self, samples_matrix, labels_matrix):
        ''' This is the general fitness evaluation function that takes as input a set of noisy samples and their corres-
        ponding set of correct labels in order to determine a fitness value of the individual. This function relies on
        the nc_predict and nc_case_fitness functions, so it is meant to be used during training with mini patches. It is
        not called nc_fitness because population dynamics rely on the function being called fitness. For a convolutional
        fitness equivalent, see score function.'''

        batch_size = len(samples_matrix)

        case_fitnesses = []

        for i in range(batch_size):
            case_fitnesses.append(self.nc_case_fitness(samples_matrix[i], labels_matrix[i]))

        self.fitness_value = np.asarray(case_fitnesses).mean()

        return self.fitness_value

    def predict(self, instance):
        ''' This is the predict function meant to be used during testing (i.e. when evaluating the filter on regular
        sized images and not on minipatches). This predict functions does applies the individual's tree in a convolutional
        fashion to whatever instance is given, in order to generate a full reconstruction/output/clean image/residual
        prediction.'''

        # Uses scipy generic filter method to perform the convolution-alike slide
        result = ndimage.generic_filter(instance, self.tree.evaluate, size=(self.input_window, self.input_window), mode='mirror')

        return result

    def case_fitness(self, instance, label):
        ''' This is the case fitness evaluation function that relies on the predict (convolutional) function, meant for
        testing purposes only (i.e. not traning).'''

        # Filter noisy sample (or extract noise mask)
        output = self.predict(instance)

        # Test for Numpy MSE
        result = ((label - output) ** 2).mean()

        return result

    def score(self, samples_matrix, labels_matrix):
        ''' This is the convolutional fitness evaluation function equivalent. This function relies on case_fitness and
        predict functions instead of their nc versions, so this actually employs convolution. Also, when calling this
        function, the fitness_value of the individuals is not modified / stored/ saved. Other than that, it is the same
        exact as the fitness function.'''

        batch_size = len(samples_matrix)

        case_fitnesses = []

        for i in range(batch_size):
            case_fitnesses.append(self.case_fitness(samples_matrix[i], labels_matrix[i]))

        testing_fitness = np.asarray(case_fitnesses).mean()

        return testing_fitness

    @staticmethod
    def crossover(filter1, filter2):
        ''' Same as in SimpleRegresor.'''

        upto1 = len(filter1.tree.nodes)
        upto2 = len(filter2.tree.nodes)

        # Notice Crossover protection from picking root nodes
        node1 = np.random.randint(1, upto1)
        node2 = np.random.randint(1, upto2)

        offspring1 = deepcopy(filter1)
        offspring2 = deepcopy(filter2)

        offspring1.tree, offspring2.tree = subtree_crossover(filter1.tree, node1, filter2.tree, node2)

        return offspring1, offspring2

    @staticmethod
    def mutation(filter1):
        ''' Sames as in SimpleRegresor.'''

        upto = len(filter1.tree.nodes)

        node = np.random.randint(upto)

        offspring = deepcopy(filter1)

        offspring.tree = subtree_mutation(filter1.tree, node)

        return offspring


    @staticmethod
    def protected_crossover(filter1, filter2):
        ''' Sames as in SimpleRegresor. Only Low-level primitives supported.'''

        # First offspring

        # pick node in first parent completely randomly
        node1 = np.random.randint(1, len(filter1.tree.nodes))

        # Get destiny node depth:
        origin_depth = filter1.tree.nodes[node1].current_tree_depth

        # Search for a valid subtree in second parent
        node2 = np.random.randint(1, len(filter2.tree.nodes))

        # Get subtree depth
        source_subtree_depth = filter2.tree.nodes[node2].subtree_depth

        # Verify if is a valid subtree to import
        while origin_depth + source_subtree_depth > filter1.tree.max_tree_depth:

            # Search for a valid subtree in second offspring
            node2 = np.random.randint(1, len(filter2.tree.nodes))

            # Get subtree depth
            source_subtree_depth = filter2.tree.nodes[node2].subtree_depth

        # Now perform first crossover
        offspring1 = deepcopy(filter1)

        offspring1.tree, _ = subtree_crossover(filter1.tree, node1, filter2.tree, node2)


        # Second Offspring

        # pick node in second parent completely randomly
        node2 = np.random.randint(1, len(filter2.tree.nodes))

        # Get destiny node depth:
        origin_depth = filter2.tree.nodes[node2].current_tree_depth

        # Search for a valid subtree in first parent
        node1 = np.random.randint(1, len(filter1.tree.nodes))

        # Get subtree depth
        source_subtree_depth = filter1.tree.nodes[node1].subtree_depth

        # Verify if is a valid subtree to import
        while origin_depth + source_subtree_depth > filter2.tree.max_tree_depth:

            # Search for a valid subtree in first parent
            node1 = np.random.randint(1, len(filter1.tree.nodes))

            # Get subtree depth
            source_subtree_depth = filter1.tree.nodes[node1].subtree_depth

        # Now perform first crossover

        offspring2 = deepcopy(filter2)

        _, offspring2.tree = subtree_crossover(filter1.tree, node1, filter2.tree, node2)

        return offspring1, offspring2

    @staticmethod
    def samedepths_crossover(filter1, filter2):
        ''' Sames as in SimpleRegresor. Only Low-level primitives supported.'''

        upto1 = filter1.tree.nodes[0].subtree_depth
        upto2 = filter2.tree.nodes[0].subtree_depth

        # Notice Crossover protection from picking root nodes
        depth = np.random.randint(1, min(upto1, upto2))

        # Pick candidate node for crossover point in parent 1
        node1 = np.random.randint(1, len(filter1.tree.nodes))

        # ensure is a node at the selected depth
        while filter1.tree.nodes[node1].current_tree_depth != depth:

            # Pick candidate node for crossover point in parent 1
            node1 = np.random.randint(1, len(filter1.tree.nodes))


        # Pick candidate node for crossover point in parent 2
        node2 = np.random.randint(1, len(filter2.tree.nodes))

        # ensure is a node at the selected depth
        while filter2.tree.nodes[node2].current_tree_depth != depth:

            # Pick candidate node for crossover point in parent 2
            node2 = np.random.randint(1, len(filter2.tree.nodes))

        offspring1 = deepcopy(filter1)
        offspring2 = deepcopy(filter2)

        offspring1.tree, offspring2.tree = subtree_crossover(filter1.tree, node1, filter2.tree, node2)

        return offspring1, offspring2
