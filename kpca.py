#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings

class KPCA:
    def __init__(self, X, kernel, d):
        """
        KPCA object
        Inputs:
        
        X: dxn matrix
        kernel: kernel function from kernel class
        d: number of principal components to be chosen
        """
        self.X = X
        self.kernel = kernel 
        self.d = d
    
    def _is_pos_semidef(self, x):
        return np.all(x >= 0)

    def __kernel_matrix(self):
        """
        Compute kernel matrix
        Output:
        
        K: nxn matrix
        """
        K = []
        r, c = self.X.shape
        for fil in range(c):
            k_aux = []
            for col in range(c):
                k_aux.append(self.kernel(self.X[:, fil], self.X[:, col]))
            K.append(k_aux)
        K = np.array(K)
        # Centering K
        ones = np.ones(K.shape)/c
        K = K - ones@K - K@ones + ones@K@ones
        return K
    
    def __descomp(self):
        """
        Decomposition of K
        Output:
        
        tuplas_eig: List of ordered tuples by singular 
                    values; (singular_value, eigenvector)
        """
        self.K = self.__kernel_matrix()
        eigval, eigvec = np.linalg.eig(self.K)
        if not self._is_pos_semidef(eigval):
            warnings.warn("La matriz K no es semidefinida positiva")
        # Normalize eigenvectors and compute singular values of K
        tuplas_eig = [(np.sqrt(eigval[i]), eigvec[:,i]/np.sqrt(eigval[i]) ) for i in range(len(eigval))]
        tuplas_eig.sort(key=lambda x: x[0], reverse=True)
        return tuplas_eig
    
    def project(self):
        """
        Compute scores
        Output:
        
        scores: T = sigma * V_d^t
        """
        self.tuplas_eig = self.__descomp()
        tuplas_eig_dim = self.tuplas_eig[:self.d]
        self.sigma = np.diag([i[0] for i in tuplas_eig_dim])
        self.v = np.array([list(j[1]) for j in tuplas_eig_dim]).T
        self.sigma = np.real_if_close(self.sigma, tol=1)
        self.v = np.real_if_close(self.v, tol=1)
        self.scores = self.sigma @ self.v.T
        return self.scores
    
    def plot_singular_values(self):
        eig_plot = [np.real_if_close(e, tol=1) for (e, _) in self.tuplas_eig if e > 0.01]
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(15,7.5))
        plt.plot(list(range(1, len(eig_plot) + 1)), eig_plot)
        plt.grid(True)
        plt.title('Valores singulares de la matriz $K$ distintos de 0')
        plt.ylabel('$\sigma^2$')
        plt.show()
        
    def plot_scores_2d(self, colors, grid = True, dim_1 = 1, dim_2 = 2):
        if self.d < 2:
            warnings.warn("No hay suficientes componentes prinicpales")
            return
        
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(15,10))
        plt.axhline(c = 'black', alpha = 0.2)
        plt.axvline(c = 'black', alpha = 0.2)
        plt.scatter(self.scores[dim_1 - 1,:], self.scores[dim_2 - 1,:], c = colors)
        plt.grid(grid)
        plt.title('KPCA Space')
        plt.xlabel('${}^a$ componente principal en el espacio $\phi(X)$'.format(dim_1))
        plt.ylabel('${}^a$ componente principal en el espacio $\phi(X)$'.format(dim_2))
        plt.show()
        
    def plot_scores_3d(self, colors, grid = True, dim_1 = 1, dim_2 = 2, dim_3 = 3):
        if self.d < 3:
            warnings.warn("No hay suficientes componentes prinicpales")
            return
        
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(self.scores[dim_1 - 1,:], self.scores[dim_2 - 1,:], self.scores[dim_3 - 1,:], c = colors)
        plt.grid(grid)
        ax.axis('on')
        plt.title('KPCA Space')
        ax.set_xlabel('${}^a$ componente principal en el espacio $\phi(X)$'.format(dim_1))
        ax.set_ylabel('${}^a$ componente principal en el espacio $\phi(X)$'.format(dim_2))
        ax.set_zlabel('${}^a$ componente principal en el espacio $\phi(X)$'.format(dim_3))
        plt.show()
        
    def plot_density(self, labels, dim=1, grid = False):
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(15,5))
        for ele in labels.unique():
            sns.distplot(self.scores[dim - 1,:][np.where(labels == ele)], hist = False, 
                         kde = True, kde_kws = {'linewidth': 3}, label = ele)
        plt.grid(grid)
        plt.legend()
        plt.title('Distribuciones en la ${}^a$ componente principal'.format(dim))
        plt.show()