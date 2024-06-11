#-----------------------------------------------------------------------------------#
# This file contains the necessary methods to perform typical GP operations, e.g.   #
# subtree crossover, subtree mutation, and point mutation.                          #
#
# Some of these methods however, are implemented only at an rudimentary level. For  #
# example, subtree crossover is not protected in anyway (can generate gramatically  #
# invalid trees, or trees that exceed maximum tree depth), and subtree mutation re- #
# quires as argument the node from which a new subtree must replace the current one #
# (i.e., instead of randomly choosing such node itself). These tasks need to be     #
# performed by upper-level methods associated to GP individuals that make calls to  #
# the functions herein defined. For an example of such implementation, refer to the #
# SimpleRegressor class that defines a GP individual where the methods here defined #
# are called by other similarly named 'crossover' and 'mutation' methods that encap-#
# sulate them, and perform some additional grammatical verifications.               #
#
# This file is part of TurboGP, a Python library for Genetic Programming (Koza,1992)#
# by Rodriguez-Coayahuitl, Morales-Reyes, HJ Escalante. 2020.                       #
# Instituto Nacional de Astrofisica, Optica y Electronica (INAOE). Puebla, Mexico   #
# Development funded by CONACyT grant No. 436184, CONCyTEP grant 2019-52D, and INAOE#
# Distributed under GNU General Public License                                      #
#-----------------------------------------------------------------------------------#


from inspect import signature
from copy import deepcopy
from GPIndividuals import *
from LowLevel import *
from Mezzanine import *
from HighLevel import *
import numpy as np
import random

def fastNodeCopy(node):
    '''Auxiliary function to implement subtree crossovers (replaces deepcopy for nodes, which is super slow)'''

    a = Node(node_id = node.node_id,
             node_list = None,
             current_tree_depth = node.current_tree_depth,
             max_tree_depth = node.max_tree_depth,
             i1_set = node.i1_set,
             parent_type = node.parent_type,
             high_depth_allocation = node.high_depth_allocation,
             grow_method = node.grow_method,
             force_type ='other')

    a.children_id = node.children_id
    a.node_type = node.node_type
    a.function = node.function
    a.number_of_inputs = node.number_of_inputs

    return a

def subtree_walk(tree, from_node, subtree):
    '''Auxiliary function to implement subtree crossovers'''

    subtree.append(fastNodeCopy(tree.nodes[from_node]))
    for child in tree.nodes[from_node].children_id:
        subtree_walk(tree, child,subtree)

def tree_walk(tree, current_node, up_to_node, subtree):
    '''Auxiliary function to implement subtree crossovers'''

    subtree.append(fastNodeCopy(tree.nodes[current_node]))
    for child in tree.nodes[current_node].children_id:
        if child != up_to_node:
            tree_walk(tree, child,up_to_node,subtree)

def rebrand_tree_nodes(tree,breaknode):
    '''Auxiliary function to implement subtree crossovers'''

    old_nodeid = {}
    new_nodeid = 0

    for node in tree:
        old_nodeid[node.node_id] = new_nodeid
        node.node_id = new_nodeid
        new_nodeid += 1

    old_nodeid[breaknode] = new_nodeid

    for node in tree:
        new_child = []
        for child in node.children_id:
            new_child.append(old_nodeid[child])
        node.children_id = new_child

    return tree, new_nodeid

def rebrand_subtree_nodes(tree,rootnode):
    '''Auxiliary function to implement subtree crossovers'''

    old_nodeid = {}
    new_nodeid = rootnode

    for node in tree:
        old_nodeid[node.node_id] = new_nodeid
        node.node_id = new_nodeid
        new_nodeid += 1

    #for node in tree:
    #    for child in node.children_id:
    #        child = old_nodeid[child]
    for node in tree:
        new_child = []
        for child in node.children_id:
            new_child.append(old_nodeid[child])
        node.children_id = new_child

    return tree


def update_tree_depth(node, new_depth):
    '''This functions is used to update the depth of every node in a tree. It's necessary to call it on offsprings
    generated by GP tree operations such as subtree crossover or subtree mutation. It is necessary to keep updated the
    current depth of every node in a tree in order to perform protected mutations and crossovers, that do not generate
    offsprings that grow in size beyond that of the max tree depth allowed. Notice this function performs a very
    different task than update_subtree_depth, even though they are similar in name, should not be confused.'''

    node.current_tree_depth = new_depth
    if node.children_id:
        for child in node.children_id:
            update_tree_depth(node.node_list[child], new_depth+1)

    return None



def subtree_crossover(tree1, node1, tree2, node2):
    '''This function is used to perform the typical GP crossover, i.e. subtree crossover. Receives as input two trees
    and the desired nodes IDs (integers) for each tree. Returns as output two new trees.

    It's important to remark that this subtree_crossover method is not protected in three ways:

    1. When performing crossover between two trees the points of crossing might be such that one or both of the
    resulting offspring trees is deeper than the original max_depth parameter with which the initial population was
    created. Contrast it with subtree_mutation which IS protected.

    2. There is no grammatical verification for trees generated with subtree crossover at this level of the code, that
    is, high level nodes may end up connected to low level nodes (producing a type error), or low level nodes may be
    connected as childred nodes of mezzanine nodes.

    3. Last one is more kind of a bug: if either node1 and/or node2 are 0 (root node), the resulting offspring will be a
    forest composed of two (disconnected) trees with only the original tree being functional.

    Individuals must define a subtree operation that relies on this method but implement all measures to avoid points
    (2), (3), and -only if desired- (1); see SimpleRegressor for an example of such implementations.
    '''
    #First offspring================================
    # The tree that keeps its root
    tree = []
    tree_walk(tree1,0,node1,tree)
    # The subtree of tree2 rooted ad node2 that will stick in tree1 at node1
    subtree = []
    subtree_walk(tree2,node2,subtree)
    # Rearrange tree & subtree id numbers
    new_tree, subtree_new_root =  rebrand_tree_nodes(tree,node1)
    new_tree2 = rebrand_subtree_nodes(subtree,subtree_new_root)
    # Create empty tree
    offspring1 = Tree(tree1.max_tree_depth, tree1.i1_set, grow_method='variable', high_depth_allocation=tree1.high_depth_allocation)
    # fusion tree and subtree
    offspring1.nodes = new_tree + new_tree2
    # Aknowledge nodes of their new tree
    offspring1.refresh_node_list()

    #Second offspring===============================
    # Now tree2 keeps its root
    tree = []
    tree_walk(tree2,0,node2,tree)
    # And the subtree rooted at node1 of tree1 will be glued in tree2 at node2
    subtree = []
    subtree_walk(tree1,node1,subtree)
    # Rearrange tree & subtree id numbers
    new_tree, subtree_new_root =  rebrand_tree_nodes(tree,node2)
    new_tree2 = rebrand_subtree_nodes(subtree,subtree_new_root)
    # Create empty tree
    offspring2 = Tree(tree2.max_tree_depth, tree2.i1_set, grow_method='variable', high_depth_allocation=tree2.high_depth_allocation)
    # fusion tree and subtree
    offspring2.nodes = new_tree + new_tree2
    # Aknowledge nodes of their new tree
    offspring2.refresh_node_list()

    update_tree_depth(offspring1.nodes[0],0)
    update_tree_depth(offspring2.nodes[0],0)

    offspring1.update_subtree_depth(offspring1.nodes[0])
    offspring2.update_subtree_depth(offspring2.nodes[0])

    return offspring1, offspring2

def point_mutation(tree, node):
    '''Point Mutation a.k.a. Node Mutation or simply Mutation. Receives as input a tree and a selected node to be
    replaced with a compatible primitive. Returns a new tree with the node change applied.

    It also semi-respects the primitive class (lowlevel, highleve, mezzanine, input, constant, vector input...);
    it only chooses a primitive within the same class, if input based, but it might switch it from a variable to a
    constant type, if leaf node is selected.'''


    new_tree = deepcopy(tree)
    new_tree.refresh_node_list()
    new_tree.update_subtree_depth(new_tree.nodes[0])

    if new_tree.nodes[node].node_type == 'f1':
        # Mutate to an number-of-inputs-compatible primitive
        candidate_function_inputs = 0
        while candidate_function_inputs != new_tree.nodes[node].number_of_inputs:
            candidate_function = np.random.choice(a=Node.f1_set)
            candidate_function_inputs = len(signature(globals()[candidate_function]).parameters)
        new_tree.nodes[node].function = candidate_function

    if new_tree.nodes[node].node_type == 'f2':
        # Mutate to (possibly) another Mezzanine function; all Mezzanine functions are unary
        candidate_function = np.random.choice(a=Node.f2_set)
        new_tree.nodes[node].function = candidate_function

    if new_tree.nodes[node].node_type == 'f3':
        # Mutate to an number-of-inputs-compatible primitive
        candidate_function_inputs = 0
        while candidate_function_inputs != new_tree.nodes[node].number_of_inputs:
            candidate_function = np.random.choice(a=Node.f3_set)
            candidate_function_inputs = len(signature(globals()[candidate_function]).parameters)
        new_tree.nodes[node].function = candidate_function

    if new_tree.nodes[node].node_type == 'i1' or new_tree.nodes[node].node_type == 'i2':
        # Mutate to scalar inputs/features or constants
        new_tree.nodes[node].node_type=np.random.choice(a=['i1', 'i2'])
        if new_tree.nodes[node].node_type == 'i1':
            # Pick a random feature/input NOTICE how it respects the initial set of valid features setup when the tree was first created
            new_tree.nodes[node].function = np.random.choice(tree.i1_set)
        if new_tree.nodes[node].node_type == 'i2':
            # Generate random between i2_set[0] and i2_set[1]
            new_tree.nodes[node].function = np.random.uniform(Node.i2_set[0],Node.i2_set[1])

    if new_tree.nodes[node].node_type == 'i3':
        # Pick another trimmer
        new_tree.nodes[node].function = np.random.choice(new_tree.nodes[node].i3_set)

    if new_tree.nodes[node].node_type == 'i4':
        # Pick another mask
        new_tree.nodes[node].function = np.random.choice(len(Node.i4_set))


    return new_tree

def subtree_mutation(tree,node):
    '''Subtree mutation receives as input a tree and an ID of one of its nodes. The subtree rooted at such node gets
    replaced with a new, randomly generated, (sub)tree. It is one of the two standard operations, along subtree cross-
    over. Unlike the subtree_crossover method defined above, however, subtree_mutation is protected in every sense:
    (1) it always  generates offspring that respects the max tree depth; (2) it always generates trees that are
    gramatically correct, and (3) if the node of one of the trees is the root, then such tree gets replaced by a
    completely new, randomly generated, tree. This function internally works by relying on the subtree_crossover method
    previously defined (it works by generating a new random tree and performing crossover between the tree at the node
    selected and the random tree at the root, and discarding the other tree generated); so it can serve as a template on
    how to define functions that rely on subtree_crossover while generating always valid offsprings. '''

    # Determine the max allowable depth for the new subtree based on the
    #depth of the node to be replaced and the initial max allowable depth
    max_depth = tree.max_tree_depth - tree.nodes[node].current_tree_depth

    #-----------------------------------------------------------------------------------
    # if the node picked is at the max allowed depth, then subtree mutation
    # shall call point mutation in order to avoid making the tree deeper.
    if max_depth <= 0:

        result = point_mutation(tree, node)
        return result

    #-----------------------------------------------------------------------------------
    # if the node picked is the root, then subtree mutation
    # shall generate an entirely new tree unrelated to the parent tree.
    if node == 0:

        result = Tree(tree.max_tree_depth, tree.i1_set, grow_method='variable', high_depth_allocation=tree.high_depth_allocation)
        result.grow_random_tree()
        return result

    #-----------------------------------------------------------------------------------

    # Generate a new tree; but verify what kind of root node it must have according to the root node of the subtree that
    # will replace.

    # If No Mezzanine functions defined, then allow only low level root type
    if not Node.f2_set:
        subtree = Tree(tree.max_tree_depth, tree.i1_set, grow_method='variable', high_depth_allocation=tree.high_depth_allocation, force_root='None')
    else:
        # Otherwise, allow other types of nodes
        if tree.nodes[node].node_type == 'f2':
            # if Mezzanine, then root node might be either low level or mezzanine
            root_type = np.random.choice(a=['None', 'f2'])
            subtree = Tree(tree.max_tree_depth, tree.i1_set, grow_method='variable', high_depth_allocation=tree.high_depth_allocation, force_root=root_type)
        elif tree.nodes[node].node_type == 'f3' or tree.nodes[node].node_type == 'i3' or tree.nodes[node].node_type == 'i4':
            # if High Level or Array input/const, then root node has to be high Level
            subtree = Tree(tree.max_tree_depth, tree.i1_set, grow_method='variable', high_depth_allocation=tree.high_depth_allocation, force_root='f3')
        else:
            # if Low level function or scalar input/constant, then root can be low level or mezzanine too:
            root_type = np.random.choice(a=['None', 'f2'])
            subtree = Tree(tree.max_tree_depth, tree.i1_set, grow_method='variable', high_depth_allocation=tree.high_depth_allocation, force_root=root_type)

    subtree.grow_random_tree(real_depth = tree.nodes[node].current_tree_depth, parent_type = tree.nodes[node].parent_type)

    # Now we can actually use subtree_crossover operation
    result, _ = subtree_crossover(tree,node,subtree,0)

    # Make sure max_tree_depth is inherited to child
    result.max_tree_depth = tree.max_tree_depth

    # Update the depth of each node for further processings.
    update_tree_depth(result.nodes[0],0)

    return result

def numeric_mutation(tree, fitness_value, c=0.02):
    ''' Numeric Mutation was proposed by Matthew Evett and Thomas Fernandez (1998) as a genetic operation 
    aimed at optimized the numerical constants (i2-type of nodes, under Rodriguez-Coayahuitl taxonomy, IEEE-
    ROPEC-2019) of GP individuals. The idea is to perturb every constant leaf node value with an additive
    factor taken from a range specified by the error of the best indiviual in the population, multuplied by
    a constant factor. Evett & Fdez suggest a value of 0.02 for the constant factor they called "temperature
    variance constant". The idea is that, as the population moves towards better solutions, the perturbation
    added to the constant nodes will decrease, whereas in early stages of an evolutionary process, this GP
    operation is more aggressive, in a similar fashion to the concept of simmulated annealing.
    
    This method implements the basic logic for numeric mutation; nevertheless, it needs to be complemented
    by corresponding wrapper functions in GP individuals classes definitions.'''
    
    # Calc range for allowed perturbation
    temp_factor = fitness_value * c
    
    rng = np.random.default_rng()
    
    # Sweep all tree nodes searching for i2 nodes
    for node in tree.nodes:
        # if scalar constant
        if node.node_type == 'i2':
            # generate additve factor
            #noise = np.random.uniform(-temp_factor, temp_factor)
            noise = (2 * temp_factor) * rng.random() - temp_factor
            # modify node value
            node.function = node.function + noise
    
    return tree

def function_composition(tree, g_depth, grow_method='variable'):
    ''' Composition is a new kind of GP operation, a variant of mutation. This operations receives
    as input a tree and a control parameter named g_depth. A new tree , g, is created with maximum
    depth g_depth. A subtree of tree g (preferably one of its leaf nodes) gets replaced by the whole
    input tree (by replacing the root of the subtree being replaced with the root of input tree).

    This operation can be thought as the opposing or inverse operation to subtree mutation. In
    subtree mutation a tree gets disrupted by replacing on of its subtrees with a random grown one,
    whereas in compisition is the random tree the one that gets disrupted. But the main idea behind
    composition is not to disrupt anyone, but instead to complement an already good function.'''

    # TODO: The logics to (preferably) pick one of the leaf or deepest nodes in G for insertion
    # point are still not implemented.

    # Create new random tree that represents function g
    g = Tree(g_depth, tree.i1_set, grow_method='variable', high_depth_allocation=tree.high_depth_allocation)
    g.grow_random_tree()

    # Pick some random point for insertion (read TODO)
    upto = len(g.nodes)
    node = np.random.randint(1, upto)

    # Insert input tree into some random point of tree g

    _, result = subtree_crossover(tree, 0, g, node)

    # Make sure max_tree_depth is updated according to how much the tree grew
    result.update_subtree_depth(result.nodes[0])
    result.max_tree_depth = result.nodes[0].subtree_depth

    # Update the depth of each node for further processings.
    update_tree_depth(result.nodes[0],0)

    return result

def forest_crossover(forest1, tree1, forest2, tree2):
    '''This is a simple form of forest crossover
    Two trees are exchanged between two forests'''

    offspring1 = deepcopy(forest1)
    offspring2 = deepcopy(forest2)

    offspring1.trees[tree1] = deepcopy(forest2.trees[tree2])
    offspring2.trees[tree2] = deepcopy(forest1.trees[tree1])

    return offspring1, offspring2

def forest_mutation(forest, tree, grow_method='variable', tree_depth=0):
    '''This is a simple form of forest mutation
    A tree in the forest gets replaces by a new random tree of max depth tree_depth
    if no tree_depth is speficied, the max depth of the replace tree is used'''

    if tree_depth == 0:
        tree_depth = forest.trees[tree].max_tree_depth

    new_forest = deepcopy(forest)
    # Generate a new tree; but respect the variable/feature set i1 of the forest that is being mutated.
    new_tree = Tree(tree_depth,forest.i1_set,grow_method)
    new_tree.grow_random_tree()

    new_forest.trees[tree] = new_tree

    return new_forest

def forest_GA_crossover(forest1, forest2, co_point):
    '''This is a simple Genetic Algorithm-like type of crossover for forests.
    All Offspring1 trees upto co_point index-1 are taken from parent forest1, the rest
    i.e. from co_point and onwards, from parent forest2. And viceversa for Offspring2.'''

    offspring1 = deepcopy(forest1)
    offspring2 = deepcopy(forest2)

    for x in range (co_point, len(forest2.trees)):
        offspring1.trees[x] = deepcopy(forest2.trees[x])
        offspring2.trees[x] = deepcopy(forest1.trees[x])

    return offspring1, offspring2
