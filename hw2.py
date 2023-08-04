import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:,-1]
    s = len(labels)
    s_i = np.unique(labels, return_counts=True)[1]
    gini = 1 - np.sum( (s_i / s)**2 )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data[:,-1]
    s = len(labels)
    s_i = np.unique(labels, return_counts=True)[1]
    entropy = - np.sum( (s_i / s) * np.log2(s_i / s) )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {} # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # split the groups' data
    data_df = pd.DataFrame(data)
    for value in np.unique(data[:,feature]):
        groups[value] = data_df[data_df[feature] == value].values
    
    s_v = np.unique(data[:,feature], return_counts=True)[1]
    s = len(data)
    
    # calculate goodness of split
    goodness = impurity_func(data)
    for i in range(len(s_v)):        
        goodness -= (s_v[i] / s) * impurity_func(groups[np.unique(data[:,feature])[i]])
    
    # calculate gain ratio
    if gain_ratio:
        split_info = - np.sum( (s_v / s) * np.log2(s_v / s) )
        try:
            goodness = goodness / split_info
        except RuntimeWarning:
            result = 0.0
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups

class DecisionNode:

    def __init__(self, data, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio 
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # find most common label in the data
        labels = self.data[:,-1]
        pred = np.unique(labels)[np.argmax(np.unique(labels, return_counts=True)[1])]
  
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
     
    def split(self, impurity_func):

        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.depth < self.max_depth:
            # find best feature
            best_feature = None
            split_data = None
            most_gain = 0
            for i in range(len(self.data[0]) - 1):
                gain, groups = goodness_of_split(self.data, i, impurity_func, self.gain_ratio)
                if gain > most_gain:
                    most_gain = gain
                    best_feature = i
                    split_data = groups
            
#             if can't improve, set to be a leaf
            if most_gain == 0:
                self.terminal = True

            
            # calculate chi square value
            elif self.chi in chi_table[1].keys():
                chi_square = 0
                labels = self.data[:,-1]
                p_labels = labels[labels=='p']
                e_labels = labels[labels=='e']
                P_p = len(p_labels) / len(labels)
                P_e = len(e_labels) / len(labels)

                for sub_data in split_data.values():
                    D = len(sub_data)
                    sub_labels = np.unique(sub_data[:,-1], return_counts=True)
                    for i in range(len(sub_labels[0])):
                        if sub_labels[0][i] == 'p':
                            p = sub_labels[1][i]
                            chi_square += (p - D * P_p)**2 / (D * P_p)
                        elif sub_labels[0][i] == 'e':
                            e = sub_labels[1][i]
                            chi_square += (e - D * P_e)**2 / (D * P_e)
            
                freedom_deg = (len(split_data.keys()) - 1) 
        
                if chi_square < chi_table[freedom_deg][self.chi]:
                    self.terminal = True
            
            if not self.terminal:
                # update best_feature value
                self.feature = best_feature

                # build children nodes and update
                for key, value in split_data.items():        
                    node = DecisionNode(data=value, feature=-1, depth=self.depth+1, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)           
                    self.add_child(node, key)
                
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # initialize selected root node
    root = DecisionNode(data=data, feature=-1,depth=0, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    
    # check if the data is homogeneous
    if len(np.unique(data[:,-1])) == 1:
        root.terminal = True 
        return root
        
    # split the node
    root.split(impurity)
    
    # hold a list of all node in the tree which didnt split yet
    next_children = root.children.copy()
    
    while len(next_children) > 0:
        node = next_children.pop(0)
        node.split(impurity)
        if node.children != []:
            next_children.extend(node.children)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root

def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    instance = instance[:-1].copy()
    node = root
    
    while not node.terminal and node.children != []:
        # find next node according to split feature value of instance
        instance_val = instance[node.feature]      
        if instance_val in node.children_values:
            child_index = node.children_values.index(instance_val)
            node = node.children[child_index]
        else:
            break
    
    pred = node.pred
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred

def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    total_instances = len(dataset)
    correct_predictions = 0
    
    for instance in dataset:
        prediction = predict(node, instance)
        if prediction == instance[-1]:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_instances
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy

def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing  = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, max_depth=max_depth)
        train_ac = calc_accuracy(tree, X_train)
        test_ac = calc_accuracy(tree, X_test)
        training.append(train_ac)
        testing.append(test_ac)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):

    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, chi=chi)
        chi_training_acc.append(calc_accuracy(tree, X_train))
        chi_testing_acc.append(calc_accuracy(tree, X_test))
        depth.append(calc_depth(tree))
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth

def calc_depth(tree):
    """
    Calculate and return the depth of a given tree
    Input:
    - tree: a tree
    
    Ooutput:
    - depth: the calculated depth od the tree
    """
    children_queue = tree.children.copy()
    depth = tree.depth
    node = tree
    while len(children_queue) > 0:
        node = children_queue.pop(0)
        if node.depth > depth:
            depth = node.depth
        children_queue.extend(node.children)
    
    return depth
    

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_nodes = 1
    children_queue = node.children.copy()
    while len(children_queue) > 0:
        node = children_queue.pop(0)
        children_queue.extend(node.children)
        n_nodes +=1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






