"""
This module implements gradient descent algorithm of L2-regularized logistic regression. The method is launched on a simulated dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def convert_label(label):
    """
    Convert binary class labels to +1/-1
    :param label: target classes
    :return: +1/-1
    """
    for i in range(label.shape[0]):
        if label[i] == 1:
            label[i] = +1
        else:
            label[i] = -1
    return label


def simulate_data():
    """
    Generate simulated data
    :return: training data, test data
    """
    X, y = make_classification(n_samples=100, n_features=20, n_clusters_per_class=1)
    y = convert_label(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def objective(beta, lamda, x, y):
    """
    Compute objective of l2-regularized logistic regression
    :param beta: coefficients of features
    :param lamda: the regularization parameter
    :param x: features derived from data
    :param y: categorical output
    :return: objective value
    """
    n = len(y)
    yx = y[:, None]*x
    obj = 1/n*(np.sum(np.log(np.exp(-yx.dot(beta))+1))) + lamda*np.linalg.norm(beta)**2
    return obj


def computegrad(beta, lamda, x, y):
    """
    Compute gradient of objective function of l2-regularized logistic regression
    :param beta: coefficients of features
    :param lamda: the regularization parameter
    :param x: features derived from data
    :param y: categorical output 
    :return: gradient of objective
    """
    n = len(y)
    yx = y[:, None] * x
    upper = yx * np.exp(-yx.dot(beta[:, None]))
    bottom = np.exp(-yx.dot(beta)) + 1
    gradient = -1 / n * np.sum(upper / bottom[:, None], axis=0) + 2 * lamda * beta
    return gradient


def backtracking(beta, lamda, x, y, t=1, alpha=0.5, beta_s=0.8, max_iter=100):
    """
    calculate step size of gradient descent algorithm
    :param beta: coefficients of features
    :param lamda: the regularization parameter
    :param x: Current point
    :param y: categorical output 
    :param t: Starting (maximum) step size
    :param alpha: Constant used to define sufficient decrease condition
    :param beta_s: Fraction by which we decrease t if the previous t doesn't work
    :param max_iter: Maximum number of iterations to run the algorithm
    :return: Matrix of estimated x's at each iteration, with the most recent values in the last row.
    """
    grad_beta = computegrad(beta, lamda, x=x, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_t = 0
    iter = 0
    while (found_t == 0 and iter < max_iter):
        if (objective(beta - t*grad_beta, lamda, x=x, y=y)) < (objective(beta, lamda, x=x, y=y)-alpha*t*(norm_grad_beta)**2):
            found_t = 1
        elif(iter == max_iter):
            stop("Maximum number of iterations reached")
        else:
            t = t*beta_s
            iter = iter + 1
    return t


def graddescent(beta_init, lamda, x, y, max_iter=1000):
    """
    Run gradient descent with backtracking line search
    :param beta_init: Starting coefficients
    :param lamda: the regularization parameter
    :param x: Current point
    :param y: categorical output 
    :param max_iter: Maximum number of iterations to perform
    :return: Matrix of estimated x's at each iteration, with the most recent values in the last row.
    """
    beta = beta_init
    grad_beta = computegrad(beta, lamda, x=x, y=y)
    beta_vals = beta
    iter = 0
    while (iter < max_iter):
        t = backtracking(beta, lamda, x=x, y=y)
        beta = beta - t * grad_beta
        beta_vals = np.vstack((beta_vals, beta))
        grad_beta = computegrad(beta, lamda, x=x, y=y)
        iter = iter + 1
    return beta_vals


def objective_plot(betas_gd, lamda, x, y):
    """
    Plot objective value of each iteration
    :param betas_gd: Matrix of estimated x's at each iteration, with the most recent values in the last row.
    :param lamda: the regularization parameter
    :param x: Current point
    :param y: categorical output 
    :return: plot of objective vs. iterations
    """
    num = np.size(betas_gd, 0)
    objs_gd = np.zeros(num)
    for i in range(0, num):
        objs_gd[i] = objective(betas_gd[i], lamda, x=x, y=y)
    plt.plot(range(1, num + 1), objs_gd, label='gradient descent')
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.show()
    plt.interactive(False)


def compute_misclassification_error(beta_opt, x, y):
    """
    Compute misclassification error
    :param beta_opt: Matrix of estimated x's at each iteration, with the most recent values in the last row
    :param x: training data of x
    :param y: test data of y
    :return: misclassification error
    """
    y_pred = 1/(1+np.exp(-x.dot(beta_opt))) > 0.5
    y_pred = y_pred*2 - 1  # Convert to +/- 1
    return np.mean(y_pred != y)


def plot_misclassification_error(betas_grad, x, y):
    """
    plot misclassification error vs iterations
    :param betas_grad: Matrix of estimated x's at each iteration, with the most recent values in the last row.
    :param x: training data of x
    :param y: test data of y
    :return: a plot of misclassification error vs iterations
    """
    niter = np.size(betas_grad, 0)
    error_grad = np.zeros(niter)
    for i in range(niter):
        error_grad[i] = compute_misclassification_error(betas_grad[i, :], x, y)
    plt.plot(range(1, niter + 1), error_grad, label='gradient descent')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification error')
    plt.show()
    plt.interactive(False)


def run():
    """
    :return: objective plot, misclassification error plot
    """
    print("Loading data...")
    x_train, x_test, y_train, y_test = simulate_data()
    d = np.size(x_train, 1)
    beta = np.zeros(d)
    print("Running gradient descent...")
    betas_gd = graddescent(beta_init=beta, lamda=0.1, x=x_train, y=y_train)
    print("Ploting objective values...")
    objective_plot(betas_gd, lamda=0.1, x=x_train, y=y_train)
    print("Ploting misclassification error...")
    plot_misclassification_error(betas_gd, x=x_train, y=y_train)

if __name__ == '__main__':
    run()
