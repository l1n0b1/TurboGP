from GPIndividuals import *
from GPOperators import *

import scipy.ndimage as ndimage

import numpy as np

import random


class NonConvFilter:
    ''' Sames as in NonConvolutionalFilter.py, but this version supports all abstraction level functions, (low, mezzanine,
    and high level )thanks to the specially modified protected_crossover method.'''

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
        ''' This is the predict function used for training purposes. It does not convolute the tree over the input.'''

        # Simple regressor
        result = self.tree.evaluate(instance)

        return result

    def nc_case_fitness(self, instance, label):
        ''' This is the fitness evaluation function for a single instance that is used when training. It uses nc_predict
        function to attempt to perform a prediction given an input instance and then compares against the correct value
        by looking at the center pixel from the passed patch-label. The prediction can be either the correct pixel value
        (denoised) or the noise level (residual approach), this will depend on the kind of labels passed for training.'''

        # Filter central pixel
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
        sized images and not on mini-patches). This predict functions does applies the individuals syntaxt tree in a con-
        volutional fashion to whatever instance is given, in order to generate a full reconstruction/output/clean image/
        residual prediction.'''

        # Uses scipy generic filter method to perform the convolution-alike slide
        result = ndimage.generic_filter(instance, self.tree.evaluate, size=(self.input_window, self.input_window), mode='mirror')

        return result

    def case_fitness(self, instance, label):
        ''' This is the case fitness evaluation function that relies on the predic (non-nc) function, meant for testing
        purposes only (i.e. not traning).'''

        # Filter noisy sample (or extract noise mask, whatever)
        output = self.predict(instance)

        # Test for Numpy MSE
        # (change here for another loss/objective function)
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
    def mutation(filter1):
        ''' Sames as in SimpleRegresor, NonConvolutionalFilter.py and NonConvolutionalMezzanineFilter.py'''

        upto = len(filter1.tree.nodes)

        node = np.random.randint(upto)

        offspring = deepcopy(filter1)

        if filter1.tree.nodes[node].node_type == 'i3':
            # avoid subtree mutation; go for point_mutation
            offspring.tree = point_mutation(filter1.tree, node)
        else:
            offspring.tree = subtree_mutation(filter1.tree, node)

        return offspring

    @staticmethod
    def protected_crossover(filter1, filter2):
        ''' In this variant of NonConvFilter, protected_crossover supports all primitives abstracion levels, that is,
        in can be safely applied, while also enabling low level, mezzanine level, and high level primitives. Contrast it
        with the versions of this very same function found in NonConvolutionalFilte.py and NonConvolutionalMezzanineFilter.
        It also protected against producing offspring that exceeds the max depth parameter. So it is protected in all
        senses. It can be used as a template for other individuals/problems that require the use of all abstracion
        level primitives.'''

        # First offspring
        while True:

            max_leaf = True
            # Do not pick leafs at max depth
            while max_leaf:
                # pick node in first parent completely randomly
                node1 = np.random.randint(1, len(filter1.tree.nodes))
                # Get destiny node depth:
                origin_depth = filter1.tree.nodes[node1].current_tree_depth
                # if leaf node is not at max depth, select it
                if origin_depth != filter1.tree.max_tree_depth:
                    max_leaf = False

            # Get the node type
            origin_type = filter1.tree.nodes[node1].node_type

            # Get compatible types
            if origin_type in ['f1', 'f2', 'i1', 'i2']:
                # Scalar outputting nodes are compatible among them
                source_type = ['f1', 'f2', 'i1', 'i2']

            if origin_type in ['f3', 'i3']:
                # Array outputting nodes are partially compatible among them
                source_type = ['f3', 'i3']

            if origin_type in ['i4']:
                # Convolutional masks and const array modifiers only exchange between them
                source_type = ['i4']

            # Search for a valid subtree in second parent
            node2 = np.random.randint(1, len(filter2.tree.nodes))

            # Get subtree depth
            source_subtree_depth = filter2.tree.nodes[node2].subtree_depth

            # Verify if is a valid subtree to import
            if (origin_depth + source_subtree_depth <= filter1.tree.max_tree_depth) and (filter2.tree.nodes[node2].node_type in source_type):
                # Found valid protected crossover point
                break

        # Now perform first crossover
        offspring1 = deepcopy(filter1)
        offspring1.tree, _ = subtree_crossover(filter1.tree, node1, filter2.tree, node2)


        # Second Offspring
        while True:

            max_leaf = True
            # Do not pick leafs at max depth
            while max_leaf:
                # pick node in second parent completely randomly
                node2 = np.random.randint(1, len(filter2.tree.nodes))
                # Get destiny node depth:
                origin_depth = filter2.tree.nodes[node2].current_tree_depth
                # if leaf node is not at max depth, select it
                if origin_depth != filter2.tree.max_tree_depth:
                    max_leaf = False

            # Get the node type
            origin_type = filter2.tree.nodes[node2].node_type

            # Get compatible types
            if origin_type in ['f1', 'f2', 'i1', 'i2']:
                # Scalar outputting nodes are compatible among them
                source_type = ['f1', 'f2', 'i1', 'i2']

            if origin_type in ['f3', 'i3']:
                # Array outputting nodes are partially compatible among them
                source_type = ['f3', 'i3']

            if origin_type in ['i4']:
                # Convolutional masks and const array modifiers only exchange between them
                source_type = ['i4']

            # Search for a valid subtree in first parent
            node1 = np.random.randint(1, len(filter1.tree.nodes))

            # Get subtree depth
            source_subtree_depth = filter1.tree.nodes[node1].subtree_depth

            # Verify if is a valid subtree to import
            if (origin_depth + source_subtree_depth <= filter2.tree.max_tree_depth) and (filter1.tree.nodes[node1].node_type in source_type):
                # Found valid protected crossover point
                break

        # Now perform second crossover
        offspring2 = deepcopy(filter2)
        _, offspring2.tree = subtree_crossover(filter1.tree, node1, filter2.tree, node2)

        return offspring1, offspring2
