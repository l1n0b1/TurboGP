#-----------------------------------------------------------------------------------#
# This file contains the core classes from which GP indivuals are built, namely:    #
# nodes, trees, and forests.                                                        #
#
# Trees consist in a list of nodes; nodes can take very different forms; nodes can  #
# be either internal (function) nodes or leaf (terminal) nodes; and within each of  #
# both of these categories, there can be even different classes of nodes, such as   #
# low level nodes, mezzanine level nodes, high level nodes, feature nodes, constant #
# nodes, etc. For a detailed explanation on this taxonomy of GP Nodes, please refer #
# to Rodriguez-Coayahuitl, Morales-Reyes & HJ Escalante, "A Comparison among        #
# Different Levels of Abstraction in Genetic Programming", in IEEE International    #
# Autumn Meeting on Power, Electronics and Computing ROPEC (2019), Ixtapa, Mexico.  #
#
# Similarly, forests consist in lists of trees. GP Forest are typically used for GP #
# applications where a single scalar output is not enough, e.g. vector regression   #
# multiclass classification, and other more exotic application such as Autoencoders #
# synthesis (Evolving autoencoding structures through genetic programming, Rodriguez-
# Coayahuit, et al. in Genetic Programming and Evolvable Machines, (2019)           #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, and INAOE#
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#



from inspect import signature
from copy import deepcopy
from LowLevel import *
from Mezzanine import *
from HighLevel import *
from Trimmers import *
import numpy as np
import random

class Node:
    'common base class for all types of nodes'

    # These are class variables (known as 'static' in other languages)
    # It means all shared across all instances of the class

    # These are primitives sets, separated according to the taxonomy proposed in Rodriguez-Coayahuitl, A Morales-Reyes,
    # HJ Escalante; IEEE International Autumn Meeting on Power, Electronics and Computing (2019), Ixtapa, Mexico.

    # Low level _function_ primitives (arithmetic operations, trigonometric functions and alike)
    f1_set = []
    # Mezzanine level _function_ primitives (statistical measures over groups of data, vector-to-scalar operations, etc.)
    f2_set = []
    # High level _function_ primitives (Convolutions, Poolings, matrix operations, any vector-to-vector operation.)
    f3_set = []

    # I1 is the set of possible indices to address the feature vector, i.e. the set of features
    ## i1_set = []
    # Notice that i1_set used to be a static variable, but it was changed to allow the implementation of distributed
    # GPs as well as ensembles that rely on feature-variability.

    # I2 is a "set" of possible scalars that can be used as input as zero-argument (leaf) nodes;
    # but we will actually use it as a range, the set of reals between i2_set[0] and i2_set[1]
    i2_set = []

    # I3 is the set of "trimmers", functions that return a subset or a region over the entire input space.
    i3_set = []

    # I4 is similar to I2, in the way that defines constant inputs, but for f3 functions rather than f1s. Therefore,
    # these are vector and array constants. Unlike f2, these are not defined as a range from which are generated
    # randomly, but instead they are indeed a set of numpy vectors and arrays.
    i4_set = []




    def __init__(self, node_id, node_list, current_tree_depth, max_tree_depth, i1_set, parent_type, high_depth_allocation=.0, grow_method='full', force_type='None'):
        '''Constructor of node class
        Recieves as input a node id, for the node being created; a node list, of the nodes that comprise its tree;
        the current tree depth, at which this node is; and the max allowable tree depth (this is necessary for
        automatically and recursively grow random trees); the feature set (I1 set); the node's parent type (also
        required to grow trees and to obtain gramatically valid offspring when performing genetic operations); the grow
        method (random, full depth, etc.). force_type is an internal parameter intended to be used by genetic operations.
        '''

        self.node_id = node_id
        self.node_list = node_list
        self.current_tree_depth = current_tree_depth
        self.subtree_depth = 0
        self.max_tree_depth = max_tree_depth
        self.grow_method = grow_method
        self.i1_set = i1_set
        self.parent_type = parent_type
        self.high_depth_allocation = high_depth_allocation

        # List for/of the (possible) children nodes for this node.
        self.children_id = []


        if force_type == 'None':
            # If this node belongs to a tree created from scratch (i.e., not from genetic operations), then it takes
            # the form of some _valid_ primitive (according to its parent type, the enabled primitives set, etc.)
            self.differentiation(current_tree_depth, max_tree_depth, grow_method)
        elif force_type == 'other':
            # if this node is produced by the fastNodeCopy method, then the type is considered "other" and the node is
            # initialized with most of its variables empty.
            self.node_type = None
            self.function = None
            self.number_of_inputs = None
            return
        else:
            # If this node is the root node of a tree created by subtree mutation, it may be required to take an
            # specific class, so the offspring generated is gramatically valid.
            if force_type == 'f2':
                self.node_type = 'f2'
                self.function = np.random.choice(a=Node.f2_set)
            if force_type == 'f3':
                self.node_type = 'f3'
                self.function = np.random.choice(a=Node.f3_set)


        # We check in what type of node specialized, and in by doing so, also how many children will have.
        if  self.node_type in ('f1', 'f2', 'f3'):
            # If internal node
            self.number_of_inputs = len(signature(globals()[self.function]).parameters)
        else:
            # If leaf (terminal) node
            self.number_of_inputs = 0


    def __repr__(self):
        '''Allows to print a meaninful text representation of the instance'''
        return '({}, {}, {})'.format(self.node_id, self.function, self.children_id)



    def spring(self):
        ''' This function creates the number of children nodes for its node, according to the arity of its primitive.
        It is meant to be used in a recursive fashion in order to grow trees.'''

        for _ in range(0,self.number_of_inputs):
            self.node_list.append(Node(node_id=len(self.node_list),
                                       node_list=self.node_list,
                                       current_tree_depth=self.current_tree_depth+1,
                                       max_tree_depth=self.max_tree_depth,
                                       i1_set=self.i1_set,
                                       parent_type=self.node_type,
                                       high_depth_allocation=self.high_depth_allocation,
                                       grow_method=self.grow_method))
            self.children_id.append(len(self.node_list)-1)
        return self.children_id



    def differentiation(self, current_tree_depth, max_tree_depth, grow_method):
        ''' This is a function that allow to grow trees in different random ways. This function assigns a _valid_
        primitive for its node, according to: its parent type, the tree depth at which the node is located, the max
        allowed tree depth and the desired grow_method.

        There are two main methods used to grow trees described in the literature: full (which grows a tree to the max
        allowed depth), and random (which grows a tree to a random depth). Here we propose variant of random grow we call
        'variable', which is the default method to grow trees and subtrees (for subtree mutation). Variable consists
        in randomly growing a tree such that the deeper the node is in the tree, the higher is the probability of
        taking the form of a terminal node.

        Variable grow method allows to generate populations with a diverse range of tree sizes, without having to
        manually rely on the 50/50 standard method. As such, both full and grow methods, are still not fully implemented;
        though, they should not be difficult to implement, by using variable method (which is more complex) as guide.'''

        # Max allowed depth for low level stage in the tree (scalar to scalar nodes)
        max_low_depth = int(max_tree_depth * (1.0 - self.high_depth_allocation))

        #if grow_method=='full':
        #    # If still not max depth, pick a function
        #    if current_tree_depth < max_tree_depth:
        #        # this will be eventually a probabilistic procedurally pick
        #        self.node_type='f1'
        #        # This should be done with dicts or string comprehensions
        #        if self.node_type == 'f1':
        #            self.function = np.random.choice(a=Node.f1_set)
        #        if self.node_type == 'f2':
        #            self.function = np.random.choice(a=Node.f2_set)
        #        if self.node_type == 'f3':
        #            self.function = np.random.choice(a=Node.f3_set)#This is wrong, actually: cannot pick f3 if parent is f1
        #    # if we reached max tree depth, pick a zero-argument function, i.e. input or constant
        #    else:
        #        # for the momment only scalar (crude) inputs/features or constants
        #        self.node_type=np.random.choice(a=['i1', 'i2'])
        #        if self.node_type == 'i1':
        #            # Pick a random feature/input
        #            self.function = np.random.choice(self.i1_set)
        #        if self.node_type == 'i2':
        #            # Generate random between i2_set[0] and i2_set[1]
        #            self.function = np.random.uniform(Node.i2_set[0],Node.i2_set[1])
        #        #TODO: Add rest of zero-argument functions

        #if grow_method=='grow':
        #    # Pick from any option
        #    self.node_type=np.random.choice(a=['f1', 'i1', 'i2'])  #TODO: Add rest of function types
        #    if self.node_type == 'f1':
        #        self.function = np.random.choice(a=Node.f1_set)
        #    if self.node_type == 'f2':
        #        self.function = np.random.choice(a=Node.f2_set)
        #    if self.node_type == 'f3':                             #This is wrong, actually
        #        self.function = np.random.choice(a=Node.f3_set)
        #    if self.node_type == 'i1':
        #        self.function = np.random.choice(self.i1_set)
        #    if self.node_type == 'i2':
        #        self.function = np.random.uniform(Node.i2_set[0],Node.i2_set[1])
        #    #TODO: Add rest of zero-argument functions

        if grow_method=='variable':
            # The closer you are to maximum depth, the higher the chances a terminal node is picked
            functype=np.random.choice(a=['input-based', 'zero-argument'], p=[1-(current_tree_depth/max_tree_depth), (current_tree_depth/max_tree_depth)])
            # However, if no high level functions defined, that is, this is a 'MidGP' individual, then force leaf node i3 (trimmer)
            if (Node.f3_set == None) and (self.parent_type == 'f2'):
                functype = 'zero-argument'

            if functype=='input-based':
                if self.parent_type == 'f2':
                    # Vector or matrix receiving parent_type
                    self.node_type = 'f3'
                elif self.parent_type == 'f3':
                    # Vector or matrix receiving parent_type
                    self.node_type = 'f3'
                else:
                    # scalar receiving parent
                    if Node.f2_set == None:
                        # If only low-level GP (no Mezzanine functions that allow transition to high level primitives)
                        # then force f1 type (low level primitives)
                        self.node_type = 'f1'
                    else:
                        # Meet the minimum allocation space in the tree for low level-only primitives.
                        if (current_tree_depth) < (max_low_depth):
                            self.node_type = 'f1'
                        else:
                            # After reaching threshold of low level functions-only allowance, consider Mezzanine functions
                            self.node_type = np.random.choice(a=['f1', 'f2'], p=[0.2, 0.8])
                # This should be done with dicts or string comprehensions
                if self.node_type == 'f1':
                    self.function = np.random.choice(a=Node.f1_set)
                if self.node_type == 'f2':
                    self.function = np.random.choice(a=Node.f2_set)
                if self.node_type == 'f3':
                    self.function = np.random.choice(a=Node.f3_set)


            # if we reached max tree depth, pick a zero-argument function, i.e. input or constant
            else:
                # check what type of inputs/features or constant this node needs convert to, scalar or tensorial
                if self.parent_type == 'f1':
                    # scalar receiving parent
                    self.node_type=np.random.choice(a=['i1', 'i2'], p=[0.8, 0.2])
                if self.parent_type == 'f2':
                    # vector or matric receiving parent
                    self.node_type=np.random.choice(a=['i3', 'i4'], p=[1.0, 0.0]) # This is because Mezzanine functions operate only over 1 parameter, and you want that parameter to be a set of features not a set of constants
                if self.parent_type == 'f3':
                    # vector or matric receiving parent
                    self.node_type=np.random.choice(a=['i3', 'i4'], p=[0.5, 0.5]) # This is because HighLevel functions operate over 2 parameters, a group of features and a mask (such a conv mask)
                if self.node_type == 'i1':
                    # Pick a random feature/input
                    self.function = np.random.choice(self.i1_set)
                if self.node_type == 'i2':
                    # Generate random between i2_set[0] and i2_set[1]
                    self.function = np.random.uniform(Node.i2_set[0],Node.i2_set[1])
                if self.node_type == 'i3':
                    # Pick a random feature/input
                    self.function = np.random.choice(self.i3_set)
                if self.node_type == 'i4':
                    # Pick a random item from the prototypes/masks collection
                    self.function = np.random.choice(len(Node.i4_set))



    def evaluate(self, features):
        ''' This function evaluates the node. It takes the values returned by the node's childred and applies the
        function determined by the node's primitive. Notice how this function works recursively: by calling this
        function on the root node, a complete tree can be evaluated.'''
        # if it is a zero-argument node, finish recursion and return value.
        if self.number_of_inputs == 0:
            if self.node_type == 'i1':
                return features[self.function]
            if self.node_type == 'i2':
                return self.function
            if self.node_type == 'i3':
                return globals()[self.function](features)
            if self.node_type == 'i4':
                return Node.i4_set[self.function]
        # otherwise, find out the value of the node by recursively requesting the value of its children and then
        # apply primitive.
        else:
            params = []
            for child in self.children_id:
                params.append(self.node_list[child].evaluate(features))
            return globals()[self.function](*params)


class Tree:
    'common base class for all trees'

    def __init__(self, max_tree_depth, i1_set, grow_method='variable', high_depth_allocation=.0, force_root='None'):
        '''
        The constructor simply allocates a reference to an empty list that can be used to build a tree.
        It does not automatically builds a random tree, since the empty list could be used to generate a tree
        from a crossover from other two trees.

        i1_set is an instance member. Whenever creating a tree, it is necessary to delimit the set from which terminal
        nodes may take their value.

        The function receives as input the maximum allowed tree depth, the grow_method, and the 'high_depth_allocation'
        parameter. High depth allocation parameter specifies what proportion of the tree depth will be used exclusively
        for low level nodes, allowing the rest of the tree deph to be used by high level nodes.
        '''
        self.root_type = force_root
        self.max_tree_depth = max_tree_depth
        self.grow_method = grow_method
        self.i1_set = i1_set
        self.high_depth_allocation = high_depth_allocation
        # Empty node list to represent the tree
        self.nodes = []


    def refresh_node_list(self):
        '''This function is used when the tree is the result of a genetic operator (instead of random grow).
        Nodes need to know which nodes are in their tree.'''

        for node in self.nodes:
            node.node_list = self.nodes


    def grow_random_tree(self, real_depth=0, parent_type=None):
        '''
        This function supervises the random grow of a GP tree. It fills the list of nodes that represents the tree.
        In an ideal world, a tree could or should be built recursively from the root node; however, due some technicals
        limitations in the way elements are added to lists in python, we have to use this supervisor function.
        '''
        # Create the root node
        self.nodes.append(Node(node_id=0,node_list=self.nodes,current_tree_depth=real_depth,max_tree_depth=self.max_tree_depth,i1_set=self.i1_set, parent_type=parent_type, high_depth_allocation=self.high_depth_allocation, grow_method=self.grow_method, force_type=self.root_type))
        # Create children of the root node
        children = self.nodes[0].spring()
        # Recursively create the rest of the nodes
        Tree.grow_nodes(self.nodes, children)
        self.update_subtree_depth(self.nodes[0])

    def evaluate(self, input_instance):
        ''' This function simply calls the evaluate function on the root node of the tree, then evaluation of the whole
        tree is performed recursively from the root node.'''
        #print(input_instance)
        return self.nodes[0].evaluate(features=input_instance)

    def update_subtree_depth(self, node):
        ''' This function calculates the depth of each node's subtree in a tree. Nodes need to know their subtree depth
        (the depth of the subtree rooted at them) in order to perform genetic operations with max depth protection.'''
        max_depth = 0
        if node.children_id:
            for child in node.children_id:
                depth = self.update_subtree_depth(node.node_list[child])
                if depth > max_depth:
                    max_depth = depth
            max_depth += 1
            node.subtree_depth = max_depth
            return max_depth

        else:
            node.subtree_depth = 0
            return 0

    @staticmethod
    def grow_nodes(tree, nodes_to_grow):
        'Auxiliary function to recursively grow random trees'
        if nodes_to_grow:
            for node in nodes_to_grow:
                children = tree[node].spring()
                Tree.grow_nodes(tree, children)


class Forest:
    'common base for forest-like type of GP individuals'

    def __init__(self, tree_count, max_tree_depth, i1_set, grow_method):
        '''
        This constructor automatically grows a set of tree_count amount of trees, with a max tree depth
        and using four different grow_methods: full, grow, half-half, and variable.

        i1_set is now a class member. Whenever creating a forest, it is necessary to delimit the set from which
        terminal nodes in all trees may take their value (beside constants, which is i2_set, still a static method)
        '''
        # Save the set of valid feature/variable inputs
        self.i1_set = i1_set
        # Create list of trees contained in the forest
        self.trees = []

        if grow_method in ['full', 'grow', 'variable']:
            for _ in range(tree_count):
                self.trees.append(Tree(max_tree_depth,i1_set,grow_method))
        if grow_method=='half-half':
            for _ in range(tree_count // 2):
                self.trees.append(Tree(max_tree_depth,i1_set,grow_method='full'))
                self.trees.append(Tree(max_tree_depth,i1_set,grow_method='grow'))

        for x in range(tree_count):
            self.trees[x].grow_random_tree()

    def evaluate_forest(self, input_instance):
        '''
        This function receives as input an input_instances, then passes it each of the trees that conform
        the forest and returns a numpy array with the ouput of each tree.

        TODO: To pass a different input instance to each tree.
        '''

        result = []
        for tree in self.trees:
            result.append(tree.evaluate(input_instance))

        return np.asarray(result)
