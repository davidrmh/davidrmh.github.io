#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:21:44 2020

@author: david

Implementation of Sum Product Networs (SPN)
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from jax import grad
import pandas as pd

class Node:
    '''
    This class implements a node's components
    
    Node(self, node_type, children, w = [], dist = None)
    
    Parameters:
        node_type: String with the type of node
            prod is for product nodes
            sum is for sum nodes
            leaf is for leaf nodes
            
        children: A list of Node objects.
        
        w: For sum nodes a list with the weights of each edge
        The sum of the elements must be 1 and each entry must
        be non-negative.
        
        dist: A distribution object that represents a distribution
        probability. It can be from scipy.stats module.
        This object must have implemented the following
        methods:
            pdf: Probability density
            rvs: Random variable sampling
            logpdf: Log density
            
    '''
    
    def __init__(self, node_type, children = [],
                 w = [], dist = None, name = 'node_name'):
        
        self.node_type = node_type
        self.children = children.copy()
        self.w = w.copy()
        self.dist = dist
        self.name = name
        
    def append_children(self, children = []):
        '''
        Appends children to the node

        Parameters
        ----------
        children : list of node objects to be appended as children

        Returns
        -------
        None. IN-PLACE modification
        '''
        
        self.children.extend(children)


def spn_pdf(spn, obs, w):
    '''
    Calculates the probability density of a sum
    product network

    Parameters
    ----------
    spn: A Node object. Initially this
    object represents the SPN (root node)
    
    obs : dictionary
        Dictionary with the observations for each leaf node.
        The keys are the node's name
        The values are the value of the observation
        for the corresponding leaf node.
        
    w : Dictionary with the weights of each sum node.
    See get_weigths function

    Returns
    -------
    Non negative float.
    '''
    
    node_name = spn.name
    
    #Leaf node
    if spn.node_type == 'leaf':
        #Get the observation
        #that correspond to the leaf node
        x = obs[node_name]
        
        #evaluates the observation
        #Continuous RV
        if hasattr(spn.dist, 'pdf'):
            return spn.dist.pdf(x)
        elif hasattr(spn.dist, 'pmf'):
            return spn.dist.pmf(x)
        else:
            print(f'pdf or pmf attribute missing in leaf node {node_name}')
            return None
        
    #Sum node
    elif spn.node_type == 'sum':
        
        #Gets the weights
        weights = w[node_name]
        
        #Weighted sum
        w_sum = 0.0
        
        #Number of children
        n_children = len(spn.children)
        
        #Calculates the pdf of each child
        for i in range(n_children):
            child = spn.children[i]
            w_i = weights[i]
            w_sum = w_sum + w_i * spn_pdf(child, obs, w)
        return w_sum
    
    #Product node
    elif spn.node_type == 'prod':
        prod = 1.0
        for child in spn.children:
            prod = prod * spn_pdf(child, obs, w)
        return prod
        
    
def add_children(node, sub_table):
    '''
    Helper function for the create_spn function

    Parameters
    ----------
    node : A node object
    
    sub_table : Pandas dataframe

    Returns
    -------
    None.
    IN-PLACE modification.
    Adds the children to the input *node* parameter
    '''
    
    #Iterates over the rows of the
    #filtered table
    for i in sub_table.index:
        
        child_name = sub_table.loc[i, 'child']
        child_type = sub_table.loc[i, 'child_type']
        
        if child_type == 'leaf':
            #a string
            dist = sub_table.loc[i, 'dist']
            #Removes single quotes
            #and evaluates the code
            dist = eval(dist.replace("'", ''))
            
            child = Node(child_type, 
                         name = child_name,
                         dist = dist)
            
            node.append_children([child])
        else:
            child = Node(child_type, name = child_name)
            node.append_children([child])
    return        

def find_by_name(node, name):
    '''
    Find a child node with a specific name

    Parameters
    ----------
    node : A Node object
      This could be the root node of a SPN
      
    name : String with the name of the node

    Returns
    -------
    TYPE
        A Node object
    '''
    
    if node.name == name:
        return node
    
    #List with all nodes to be
    #explored
    to_explore = node.children.copy()
    
    while to_explore != []:
        
        #FIFO search
        child = to_explore.pop(0)
        
        if child.name == name:
            return child
        
        #Adds new nodes to explore
        to_explore.extend(child.children.copy())
        
    #No node found    
    print(f'Node {name} not found')
    return None

def find_by_type(node, node_type):
    '''
    Find a set of nodes with a specified  type

    Parameters
    ----------
    node : A Node object.
        This could be the root node of a SPN
        
    node_type : string
        Type of node to be retrieved.
        'sum', 'prod' or 'leaf' nodes

    Returns
    -------
    nodes : List
        List of nodes with node_type that are children
        of *node*. If input parameter *node* is of
        *node_type* it is included in the list
    '''
    
    #To store the nodes
    nodes = []
    
    if node.node_type == node_type:
        nodes.append(node)
    
    #List with all nodes to be
    #explored
    to_explore = node.children.copy()
    
    while to_explore != []:
        
        #FIFO search
        child = to_explore.pop(0)
        
        if child.node_type == node_type:
            nodes.append(child)
        
        #Adds new nodes to explore
        to_explore.extend(child.children.copy())
        
    if nodes == []:
        print(f'No nodes of type {node_type} were found')
        return None
    return nodes
           
def create_spn(path_csv = 'spn_template.csv'):
    '''
    Creates a sum product network using a csv
    file as template

    Parameters
    ----------
    path_csv : string.
        Path to the csv file with the architecture

    Returns
    -------
    root : Node object.
        Root node of the SPN
    '''
    #opens table with the architecture
    table = pd.read_csv(path_csv)
    
    #Creates root node
    root_type = table[table['parent'] == 'root']['parent_type'][0]
    root = Node(root_type, name = 'root')
    
    #Appends root's children
    sub_table = table[table['parent'] == 'root']
    add_children(root, sub_table)
                
    #Creates the list with all the parents
    parents = list(table['parent'].unique())
    parents.remove('root')
    
    #Add the children of each parent
    while parents != []:
        
        #Table should be ordered
        target = parents.pop(0)
        
        #Filters the table
        sub_table = table[table['parent'] == target]
        
        #finds the parent
        #and adds the children
        node = find_by_name(root, target)
        add_children(node, sub_table)
    return root
    
def init_weights(spn):
    '''
    Initializes the weights of each sum node
    For a sum node, each weight will be equal
    to 1. / number of children

    Parameters
    ----------
    spn : Node representing a sum product network (root node)
    It can be a sub-tree of the spn

    Returns
    -------
    None.
    IN-PLACE modification
    '''
    
    #Finds the sum nodes
    sum_nodes = find_by_type(spn, 'sum')
    if sum_nodes == []:
        print('There are no sum nodes')
        return None
    
    for node in sum_nodes:
        n_children = len(node.children)
        weights = np.repeat(1. / n_children, n_children)
        node.w = weights
    return None    

def get_weights(spn):
    '''
    Get the weights of all sum nodes

    Parameters
    ----------
    spn : Node representing a sum product network (root node)
    It can be a sub-tree of the spn
    
    Returns
    -------
    Dictionary with keys the names of the sum nodes
    and values a numpy array with the weights
    '''
    
    #Gets the sum nodes
    sum_nodes = find_by_type(spn, 'sum')
    
    d = {}
    
    for node in sum_nodes:
        d[node.name] = node.w
    return d
    
#Mixture of gaussians
spn = create_spn('spn_template.csv')

#Initializes weights
init_weights(spn)

#Gets weights
w = get_weights(spn)

#observations for each leaf node
leaf_node = find_by_type(spn, 'leaf')
obs = {}
n_obs = 100
x_min = -10.
x_max = 10.
    
for node in leaf_node:
    obs[node.name] = np.linspace(x_min, x_max, n_obs)
    
x_axis = np.linspace(x_min, x_max, n_obs)
y_axis = spn_pdf(spn, obs, w)
plt.plot(x_axis, y_axis)

#Gradient of SPN with respect to each weight
grad_spn = grad(spn_pdf, argnums = 2)
#To store the evaluation of the gradient
#in each observation
grad_eval = []

obs_grad = {}

for i in range(n_obs):
    for node in leaf_node:
        obs_grad[node.name] = obs[node.name][i]
    grad_eval.append(grad_spn(spn, obs_grad, w))