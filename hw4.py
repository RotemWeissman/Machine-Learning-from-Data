import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):
    # Function for plotting the decision boundaries of a model
    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()


def visualize_3d(features, labels, title):
    # prepare data
    label0_data = features[labels == 0]
    label1_data = features[labels == 1]

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points for label 0
    ax.scatter(label0_data[:, 0], label0_data[:, 1], label0_data[:, 2], c='b', label='0')

    # Plot the data points for label 1
    ax.scatter(label1_data[:, 0], label1_data[:, 1], label1_data[:, 2], c='r', label='1')

    # Set labels for the axes
    ax.set_xlabel('feature 1')
    ax.set_ylabel('feature 2')
    ax.set_zlabel('feature 3')

    # Set a title for the plot
    ax.set_title(title)

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # apply bias
        X = np.c_[np.ones(X.shape[0]), X]
        y = y.reshape(-1,1)
        itr = 0
        # set initial thetas
        theta = np.random.rand(1, X.shape[1])
        # add first theta to the list
        self.thetas.append(theta)
        # first h(x)
        h = 1/(1 + np.exp(-X@theta.T))
        # compute cost
        current_cost = 1/X.shape[0] * \
                       np.sum(np.dot(-y.T, np.log(h)) - np.dot((1-y).T, np.log(1-h)))

        J = 0
        # cost improvement
        improve = current_cost - J
        
        while (itr < self.n_iter) and (improve >= self.eps):

            theta = theta - self.eta * (h-y).T@X
            h = 1/(1 + np.exp(-X@theta.T))
            J = 1/X.shape[0] * \
                np.sum(np.dot(-y.T, np.log(h)) - np.dot((1-y).T, np.log(1-h)))
            improve = current_cost - J
            current_cost = J

            self.thetas.append(theta)
            self.Js.append(J)
            
            itr += 1
        
        self.theta = theta
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.c_[np.ones(X.shape[0]),X] # apply bias
        h = 1/(1 + np.exp(-X@self.theta.T))
        preds = []
        for p in h:
            if p > 0.5:
                preds.append(1)
            else:
                preds.append(0)

        preds = np.array(preds)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    joint_data = np.c_[X,y]
    np.random.shuffle(joint_data)
    batch = len(joint_data)//folds
    accuracies = []
    for i in range(folds):
        test_idx = [xx for xx in range(i*batch,(i+1)*batch)]
        X_test = joint_data[test_idx,:-1]
        y_test = joint_data[test_idx,-1]
        train = np.delete(joint_data, test_idx, axis=0)
        X_train = train[:,:-1]
        y_train = train[:,-1]
        
        algo.fit(X_train,y_train)
        preds = algo.predict(X_test)
        correct = 0
        for j in range(len(preds)):
            if preds[j] == y_test[j]:
                correct += 1
        fold_acc = correct/len(preds)
        accuracies.append(fold_acc)
     
    cv_accuracy = sum(accuracies)/len(accuracies)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    exp = -((data - mu)**2 / (2* sigma**2))
    denom = sigma * np.sqrt(2 * np.pi)
    p = np.e ** exp / denom
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        weights = np.random.random_sample(self.k)
        self.weights = weights/weights.sum()
        split = np.array_split(data, self.k)
        self.mus = np.empty_like(self.weights)
        self.sigmas = np.empty_like(self.weights)
        for i in range(self.k):
            self.mus[i] = np.mean(split[i])
            self.sigmas[i] = np.std(split[i])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        responsibilities = self.weights * norm_pdf(data, self.mus, self.sigmas)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        self.responsibilities = responsibilities
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        weights = self.responsibilities.sum(axis=0) / len(data)
        mus = (self.responsibilities * data).sum(axis=0) / (self.weights * len(data))
        sigmas = np.sqrt(
            (1/(self.weights * len(data))) * \
            np.sum(self.responsibilities * (data - self.mus)**2, axis=0))

        self.weights, self.mus, self.sigmas = weights, mus, sigmas

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        # calculate current cost with randomized parameters
        current_cost = 0
        for j in range(self.k):
            current_cost -= np.log(np.sum(
                self.weights[j] * norm_pdf(data, self.mus[j], self.sigmas[j])
            ))
        # add current cost to the list
        self.costs = [current_cost]

        # while didn't exceed maximum iterations allowed
        for i in range(self.n_iter):
            # E-step
            self.expectation(data)
            # M-step
            self.maximization(data)
            # calculate new cost
            cost = 0
            for j in range(self.k):
                cost -= np.log(np.sum(
                    self.weights[j] * norm_pdf(data, self.mus[j], self.sigmas[j])
                ))
            # break the loop if the change is less than the tolerance
            if abs(current_cost - cost) <= self.eps:
                break
            # update
            self.costs.append(cost)
            current_cost = cost
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pdf = np.sum(weights * norm_pdf(data, mus, sigmas), axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None
        self.gmms = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # get the classes values and counts
        labels, labels_count = np.unique(y, return_counts=True)
        # get priors for each label
        self.prior = {}
        for i in range(len(labels)):
            self.prior[labels[i]] = labels_count[i] / len(y)

        # fit each data[label] with the EM class
        for label in labels:
            self.gmms[label] = {}
            data = X[y == label]
            for i in range(X.shape[1]):
                em = EM(k=self.k, random_state=self.random_state)
                em.fit(data[:, i].reshape(-1, 1))
                self.gmms[label][i] = em
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # hold list with likelihoods of each label for the data
        likelihoods = []

        # go over all possible labels
        for label, prior in self.prior.items():

            label_likelihood = 1

            # go over all features
            for i in range(len(self.gmms[label])):
                # take one feature
                x = X[:, i].reshape(-1,1)
                # extract its parameters
                weights = self.gmms[label][i].get_dist_params()[0]
                mus = self.gmms[label][i].get_dist_params()[1]
                sigmas = self.gmms[label][i].get_dist_params()[2]
                # calculate the likelihood
                likelihood = gmm_pdf(x, weights, mus, sigmas) * prior
                # multiply by the other feature likelihood
                label_likelihood *= likelihood

            # add that label likelihood to the list
            likelihoods.append(label_likelihood)

        # hold the predictions per instance
        preds = []

        # go over all instances
        for i in range(X.shape[0]):
            instance_likelihoods = []
            # go over all labels
            for j in range(len(likelihoods)):
                # get instance likelihood for this label
                instance_likelihoods.append(likelihoods[j][i])
            # find the label corresponding for the max likelihood. add to the predictions
            max_likelihood = np.argmax(instance_likelihoods)
            instance_pred = list(self.prior.keys())[max_likelihood]
            preds.append(instance_pred)

        preds = np.array(preds)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # fit logistic regression
    lor = LogisticRegressionGD(eps=best_eps, eta=best_eta)
    lor.fit(x_train, y_train)
    # predict and compute lor accuracies
    lor_prediction_x_train = lor.predict(x_train)
    lor_prediction_x_test = lor.predict(x_test)
    correct = 0
    for i in range(x_train.shape[0]):
        if lor_prediction_x_train[i] == y_train[i]:
            correct += 1
    lor_train_acc = correct/x_train.shape[0]

    correct = 0
    for i in range(x_test.shape[0]):
        if lor_prediction_x_test[i] == y_test[i]:
            correct += 1
    lor_test_acc = correct/x_test.shape[0]

    # fit naive bayes
    bayes = NaiveBayesGaussian(k=k)
    bayes.fit(x_train, y_train)
    # predict and compute naive bayes accuracies
    bayes_prediction_x_train = bayes.predict(x_train)
    bayes_prediction_x_test = bayes.predict(x_test)
    correct = 0
    for i in range(x_train.shape[0]):
        if bayes_prediction_x_train[i] == y_train[i]:
            correct += 1
    bayes_train_acc = correct/x_train.shape[0]

    correct = 0
    for i in range(x_test.shape[0]):
        if bayes_prediction_x_test[i] == y_test[i]:
            correct += 1
    bayes_test_acc = correct / x_test.shape[0]

    # plot decision regions
    plot_decision_regions(x_train, y_train, lor, resolution=0.01, title="LOR - Train Data")
    plot_decision_regions(x_train, y_train, bayes, resolution=0.01, title="Naive_Base- Train Data")
    plot_decision_regions(x_test, y_test, lor, resolution=0.01, title="LOR - Test Data")
    plot_decision_regions(x_test, y_test, bayes, resolution=0.01, title="Naive_Base- Test Data")

    plt.plot(np.arange(len(lor.Js)), lor.Js)
    plt.xscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost VS The Iteration Number For Logistic Regression Model', fontweight='bold',
              fontsize=12)
    plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    np.random.seed(1991)
    # dataset_a, naive bayes better than lor: non-linear separable data
    mus = [1, 2, 3]
    cov = np.eye(3)
    data = multivariate_normal.rvs(mus, cov, 3000)
    dataset_a_features = data[data[:, 2].argsort()]
    dataset_a_labels = np.zeros(3000)
    dataset_a_labels[0:1000] = 1
    dataset_a_labels[2500:] = 1

    # dataset_b, lor better than naive bayes: linear separable data, marginals similar
    mus_0 = [1, 1, 1]
    cov_0 = np.array([[1, 0.8, 0.8],
                      [0.8, 1, 0.8],
                      [0.8, 0.8, 1]])
    data_0 = multivariate_normal.rvs(mus_0, cov_0, 500)

    mus_1 = [-1, -1, -1]
    cov_1 = np.array([[1, -0.8, -0.8],
                     [-0.8, 1, 1],
                     [-0.8, 1, 1]])
    data_1 = multivariate_normal.rvs(mus_1, cov_1, 500)

    dataset_b_features = np.vstack((data_0, data_1))
    dataset_b_labels = np.hstack((np.zeros(500), np.ones(500)))

    # calculate and print the accuracy
    # initiate models
    lor = LogisticRegressionGD(eta=5e-05, eps=1e-06)
    bayes = NaiveBayesGaussian(k=1)

    # dataset a
    lor.fit(dataset_a_features, dataset_a_labels)
    bayes.fit(dataset_a_features, dataset_a_labels)

    lor_a_pred = lor.predict(dataset_a_features)
    bayes_a_pred = bayes.predict(dataset_a_features)

    lor_a_acc = 0
    bayes_a_acc = 0
    for i in range(len(dataset_a_labels)):
        if lor_a_pred[i] == dataset_a_labels[i]:
            lor_a_acc += 1
        if bayes_a_pred[i] == dataset_a_labels[i]:
            bayes_a_acc += 1

    lor_a_acc /= len(dataset_a_labels)
    bayes_a_acc /= len(dataset_a_labels)

    # dataset b
    lor.fit(dataset_b_features, dataset_b_labels)
    bayes.fit(dataset_b_features, dataset_b_labels)

    lor_b_pred = lor.predict(dataset_b_features)
    bayes_b_pred = bayes.predict(dataset_b_features)

    lor_b_acc = 0
    bayes_b_acc = 0
    for i in range(len(dataset_b_labels)):
        if lor_b_pred[i] == dataset_b_labels[i]:
            lor_b_acc += 1
        if bayes_b_pred[i] == dataset_b_labels[i]:
            bayes_b_acc += 1

    lor_b_acc /= len(dataset_b_labels)
    bayes_b_acc /= len(dataset_b_labels)

    # print results
    print("Dataset a:")
    print(f"LOR accuracy: {lor_a_acc}")
    print(f"Naive Bayes accuracy: {bayes_a_acc}")
    print("-----------")
    print("Dataset b:")
    print(f"LOR accuracy: {lor_b_acc}")
    print(f"Naive Bayes accuracy: {bayes_b_acc}")

    # plot 3D graphs of the datasets
    visualize_3d(dataset_a_features, dataset_a_labels, 'dataset a')
    visualize_3d(dataset_b_features, dataset_b_labels, 'dataset b')
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }



