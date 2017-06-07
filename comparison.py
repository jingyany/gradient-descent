"""
This module implements gradient descent algorithm of L2-regularized logistic regression. The method is launched on a real-world dataset, the "Spam" 
data, which can be downloaded from https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data. It is also compared with L2-regularized logistic 
regression in sklearn
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def convert_label(indicator):
    if indicator == 1:
        return +1
    else:
        return -1


def objective(beta, lamda, x, y):
    n = len(y)
    yx = y[:, None]*x
    obj = 1/n*(np.sum(np.log(np.exp(-yx.dot(beta))+1))) + lamda*np.linalg.norm(beta)**2
    return obj


def computegrad(beta, lamda, x, y):
    n = len(y)
    yx = y[:, None] * x
    upper = yx * np.exp(-yx.dot(beta[:, None]))
    bottom = np.exp(-yx.dot(beta)) + 1
    gradient = -1 / n * np.sum(upper / bottom[:, None], axis=0) + 2 * lamda * beta
    return gradient


def backtracking(beta, lamda, x, y, t=1, alpha=0.5, beta_s=0.8, max_iter=100):
    grad_beta = computegrad(beta, lamda, x=x, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_t = 0
    iter = 0
    while found_t == 0 and iter < max_iter:
        if (objective(beta - t*grad_beta, lamda, x=x, y=y)) < (objective(beta, lamda, x=x, y=y)-alpha*t*(norm_grad_beta)**2):
            found_t = 1
        elif(iter == max_iter):
            print ("Maximum number of iterations reached")
        else:
            t = t*beta_s
            iter = iter + 1
    return t


def graddescent(beta_init, lamda, x, y, max_iter=1000):
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
    num = np.size(betas_gd, 0)
    objs_gd = np.zeros(num)
    for i in range(0, num):
        objs_gd[i] = objective(betas_gd[i], lamda, x=x, y=y)
    plt.plot(range(1, num + 1), objs_gd, label='gradient descent')
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.show()
    plt.interactive(False)


def load_data():
    df = pd.read_csv('spam.csv', sep=' ',header=None)
    data = df[df.columns[0:57]]
    data_scaled = preprocessing.scale(data)
    data_scaled = pd.DataFrame(data_scaled)
    target = df[df.columns[57]]
    target = target.apply(convert_label)
    x_train, x_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=1)
    y_train = np.asarray(y_train)
    x_train = np.asarray(x_train)
    return x_train, x_test, y_train, y_test


def compute_misclassification_error(beta_opt, x, y):
    y_pred = 1/(1+np.exp(-x.dot(beta_opt))) > 0.5
    y_pred = y_pred*2 - 1  # Convert to +/- 1
    return np.mean(y_pred != y)


def plot_misclassification_error(betas_grad, x, y):
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
    :return: Optimal coefficients found using gradient descent,
             Optimal coefficients found using sklearn,
             Objective value using gradient descent,
             Objective value using sklearn
    """
    print("Loading data...")
    x_train, x_test, y_train, y_test = load_data()
    d = np.size(x_train, 1)
    beta = np.zeros(d)
    n_train = len(y_train)
    lamda = 0.1
    sklearn_lr = LogisticRegression(penalty='l2', C=1 / (2 * lamda * n_train), fit_intercept=False, tol=10e-8, max_iter=1000)
    sklearn_lr.fit(x_train, y_train)
    print("Running gradient descent...")
    betas_gd = graddescent(beta_init=beta, lamda=0.1, x=x_train, y=y_train)
    print('Optimal coefficients found using gradient descent:', betas_gd[-1, :])
    print('Optimal coefficients found using sklearn', sklearn_lr.coef_)
    print('Objective value using gradient descent:',objective(betas_gd[-1, :], lamda, x=x_train, y=y_train))
    print('Objective value using sklearn:',objective(sklearn_lr.coef_.flatten(), lamda, x=x_train, y=y_train))





if __name__ == '__main__':
    run()